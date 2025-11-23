#!/usr/bin/env Rscript
# --------------------------------------------
# Enhanced YouTube Skincare Analysis Script
# --------------------------------------------
# Features: Multi-channel/keyword search, sentiment analysis,
# comment extraction, engagement metrics, n-gram analysis
# --------------------------------------------
library(httr)
library(jsonlite)
library(tidyverse)
library(stringr)
library(tidytext)
library(lubridate)

# ========== CONFIGURATION PARAMETERS ==========

CONFIG <- list(
  # YouTube API Configuration
  api_key = Sys.getenv("AIzaSyB0MrYtkC29pfAC7oLonQk4IhHBQrzOI4g"),  # Set via environment variable or replace directly
  
  # Search Strategy: "channels", "keywords", or "both"
  search_strategy = "both",
  
  # Channel IDs to monitor (find via YouTube channel URL)
  channel_ids = c(
    "UC1o8AUVlD3A9YFqQJfHaKzA",  # Example: Hyram
    "UCzQ4q8u_z2pFqBGyBlkmfEg",  # Example: Dr. Dray
    "UCrBwXoWM_RdMBe7hK_6yijQ"   # Example: James Welsh
    # Add more channel IDs as needed
  ),
  
  # Search keywords for skincare content
  search_keywords = c(
    "skincare routine",
    "acne treatment",
    "eczema relief",
    "rosacea skincare",
    "sensitive skin",
    "hyperpigmentation treatment",
    "retinol review",
    "moisturizer review",
    "sunscreen review",
    "dermatologist skincare"
  ),
  
  # API Request Parameters
  max_results_per_query = 50,  # Max 50 per API call
  videos_per_channel = 50,
  max_comments_per_video = 100,
  
  # Time filtering
  published_after = NULL,  # NULL or date string "2024-01-01T00:00:00Z"
  published_before = NULL,
  hours_lookback = 168,  # Last week (168 hours), NULL for all time
  
  # Content filtering
  min_text_length = 15,
  min_word_count = 3,
  min_view_count = 100,
  min_engagement_rate = 0,  # Minimum (likes+comments)/views ratio
  
  # N-gram parameters
  include_unigrams = TRUE,
  include_bigrams = TRUE,
  include_trigrams = TRUE,
  top_k_results = 50,
  
  # Analysis options
  fetch_comments = TRUE,
  calculate_sentiment = TRUE,
  analyze_by_channel = TRUE,
  analyze_by_video = FALSE,
  detect_trending_topics = TRUE,
  
  # Output options
  save_outputs = TRUE,
  output_dir = "output_youtube",
  output_prefix = format(Sys.time(), "%Y%m%d_%H%M"),
  save_video_metadata = TRUE,
  save_comments = TRUE,
  save_ngrams = TRUE,
  save_summary_stats = TRUE,
  
  # API rate limiting
  api_delay_seconds = 0.5,
  respect_quota = TRUE  # Be mindful of 10,000 units/day quota
)

# Create output directory
if (CONFIG$save_outputs && !dir.exists(CONFIG$output_dir)) {
  dir.create(CONFIG$output_dir, recursive = TRUE)
}

# ========== LOGGING ==========
log_message <- function(..., level = "INFO") {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  msg <- paste0("[", timestamp, "] [", level, "] ", paste(...))
  message(msg)
  
  if (CONFIG$save_outputs) {
    log_file <- file.path(CONFIG$output_dir, 
                          paste0(CONFIG$output_prefix, "_log.txt"))
    cat(msg, "\n", file = log_file, append = TRUE)
  }
}

# ========== API KEY VALIDATION ==========
if (is.null(CONFIG$api_key) || CONFIG$api_key == "") {
  stop("YouTube API key not found. Set YOUTUBE_API_KEY environment variable or update CONFIG$api_key")
}

# ========== YOUTUBE API FUNCTIONS ==========

# Search for videos by keyword
search_videos_by_keyword <- function(keyword, max_results = 50, published_after = NULL) {
  base_url <- "https://www.googleapis.com/youtube/v3/search"
  
  params <- list(
    part = "snippet",
    q = keyword,
    type = "video",
    maxResults = min(max_results, 50),
    key = CONFIG$api_key,
    relevanceLanguage = "en",
    order = "date"
  )
  
  if (!is.null(published_after)) {
    params$publishedAfter <- published_after
  }
  
  resp <- GET(base_url, query = params)
  
  if (http_error(resp)) {
    log_message("Error searching keyword '", keyword, "': ", 
                http_status(resp)$message, level = "ERROR")
    return(tibble())
  }
  
  data <- content(resp, as = "parsed")
  
  if (length(data$items) == 0) {
    log_message("No videos found for keyword: ", keyword, level = "WARN")
    return(tibble())
  }
  
  tibble(
    video_id = map_chr(data$items, ~.x$id$videoId),
    title = map_chr(data$items, ~.x$snippet$title),
    description = map_chr(data$items, ~.x$snippet$description),
    channel_id = map_chr(data$items, ~.x$snippet$channelId),
    channel_title = map_chr(data$items, ~.x$snippet$channelTitle),
    published_at = map_chr(data$items, ~.x$snippet$publishedAt),
    search_keyword = keyword
  )
}

# Get videos from a specific channel
get_channel_videos <- function(channel_id, max_results = 50, published_after = NULL) {
  base_url <- "https://www.googleapis.com/youtube/v3/search"
  
  params <- list(
    part = "snippet",
    channelId = channel_id,
    type = "video",
    maxResults = min(max_results, 50),
    key = CONFIG$api_key,
    order = "date"
  )
  
  if (!is.null(published_after)) {
    params$publishedAfter <- published_after
  }
  
  resp <- GET(base_url, query = params)
  
  if (http_error(resp)) {
    log_message("Error fetching channel ", channel_id, ": ", 
                http_status(resp)$message, level = "ERROR")
    return(tibble())
  }
  
  data <- content(resp, as = "parsed")
  
  if (length(data$items) == 0) {
    log_message("No videos found for channel: ", channel_id, level = "WARN")
    return(tibble())
  }
  
  tibble(
    video_id = map_chr(data$items, ~.x$id$videoId),
    title = map_chr(data$items, ~.x$snippet$title),
    description = map_chr(data$items, ~.x$snippet$description),
    channel_id = map_chr(data$items, ~.x$snippet$channelId),
    channel_title = map_chr(data$items, ~.x$snippet$channelTitle),
    published_at = map_chr(data$items, ~.x$snippet$publishedAt),
    search_keyword = NA_character_
  )
}

# Get video statistics (views, likes, comments)
get_video_statistics <- function(video_ids) {
  if (length(video_ids) == 0) return(tibble())
  
  # YouTube API allows up to 50 video IDs per request
  chunks <- split(video_ids, ceiling(seq_along(video_ids) / 50))
  
  all_stats <- list()
  
  for (i in seq_along(chunks)) {
    base_url <- "https://www.googleapis.com/youtube/v3/videos"
    
    params <- list(
      part = "statistics,contentDetails",
      id = paste(chunks[[i]], collapse = ","),
      key = CONFIG$api_key
    )
    
    resp <- GET(base_url, query = params)
    
    if (http_error(resp)) {
      log_message("Error fetching video stats: ", 
                  http_status(resp)$message, level = "ERROR")
      next
    }
    
    data <- content(resp, as = "parsed")
    
    if (length(data$items) > 0) {
      stats <- tibble(
        video_id = map_chr(data$items, ~.x$id),
        view_count = map_dbl(data$items, ~as.numeric(.x$statistics$viewCount %||% 0)),
        like_count = map_dbl(data$items, ~as.numeric(.x$statistics$likeCount %||% 0)),
        comment_count = map_dbl(data$items, ~as.numeric(.x$statistics$commentCount %||% 0)),
        duration = map_chr(data$items, ~.x$contentDetails$duration %||% "")
      )
      
      all_stats[[i]] <- stats
    }
    
    Sys.sleep(CONFIG$api_delay_seconds)
  }
  
  bind_rows(all_stats)
}

# Get video comments
get_video_comments <- function(video_id, max_results = 100) {
  base_url <- "https://www.googleapis.com/youtube/v3/commentThreads"
  
  params <- list(
    part = "snippet",
    videoId = video_id,
    maxResults = min(max_results, 100),
    key = CONFIG$api_key,
    order = "relevance",
    textFormat = "plainText"
  )
  
  resp <- GET(base_url, query = params)
  
  if (http_error(resp)) {
    # Comments might be disabled
    return(tibble())
  }
  
  data <- content(resp, as = "parsed")
  
  if (length(data$items) == 0) {
    return(tibble())
  }
  
  tibble(
    video_id = video_id,
    comment_id = map_chr(data$items, ~.x$id),
    comment_text = map_chr(data$items, ~.x$snippet$topLevelComment$snippet$textDisplay),
    author = map_chr(data$items, ~.x$snippet$topLevelComment$snippet$authorDisplayName),
    like_count = map_dbl(data$items, ~as.numeric(.x$snippet$topLevelComment$snippet$likeCount %||% 0)),
    published_at = map_chr(data$items, ~.x$snippet$topLevelComment$snippet$publishedAt)
  )
}

# ========== TEXT PROCESSING ==========
data("stop_words", package = "tidytext")

custom_stopwords <- tibble(
  word = c(
    "skin", "amp", "ve", "don", "ll", "doesn", "didn", "won",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "time", "times", "video", "videos", "channel",
    "really", "just", "like", "know", "think", "one", "also",
    "get", "got", "make", "made", "use", "using", "used",
    "im", "ive", "i", "me", "my", "mine", "id", "you", "your",
    "help", "anyone", "anything", "something", "thing", "things",
    "can", "will", "would", "could", "should",
    "started", "start", "trying", "tried", "looks", "look",
    "feel", "felt", "feeling", "seem", "seems",
    "http", "www", "com", "https", "youtube", "watch"
  )
)

all_stopwords <- stop_words %>%
  select(word) %>%
  distinct() %>%
  bind_rows(custom_stopwords) %>%
  distinct()

skin_keywords <- c(
  "skin", "skincare", "eczema", "psoriasis", "rash", "rashes", "itch", "itchy",
  "flare", "flare-up", "flareups", "acne", "pimples", "blackheads",
  "whiteheads", "rosacea", "dermatitis", "hyperpigmentation", "dark spots",
  "redness", "sensitive", "barrier", "skin barrier", "spf", "sunscreen",
  "dryness", "dry skin", "oily skin", "combination skin", "scarring",
  "acne scars", "maskne", "allergic reaction", "hives", "bumps",
  "moisturizer", "cleanser", "serum", "cream", "lotion", "toner",
  "retinol", "niacinamide", "hyaluronic", "vitamin c", "salicylic",
  "routine", "dermatologist", "breakout", "pores", "texture"
)

skin_pattern <- str_c(skin_keywords, collapse = "|")

clean_text <- function(text) {
  if (is.na(text) || !is.character(text)) return("")
  text <- str_to_lower(text)
  text <- str_remove_all(text, "http\\S+|www\\.[^\\s]+")
  text <- str_remove_all(text, "@\\w+")
  text <- str_replace_all(text, "#(\\w+)", "\\1")
  text <- str_replace_all(text, "[^a-z0-9\\s]", " ")
  text <- str_squish(text)
  text
}

contains_skin_keyword <- function(text) {
  if (is.na(text) || text == "") return(FALSE)
  str_detect(text, skin_pattern)
}

# ========== MAIN PROCESSING ==========
log_message("Starting YouTube skincare analysis...")
log_message("API Key configured: ", substr(CONFIG$api_key, 1, 10), "...")

# Calculate time filter
published_after <- NULL
if (!is.null(CONFIG$hours_lookback)) {
  published_after <- format(Sys.time() - hours(CONFIG$hours_lookback), 
                            "%Y-%m-%dT%H:%M:%SZ")
  log_message("Time filter: videos published after ", published_after)
} else if (!is.null(CONFIG$published_after)) {
  published_after <- CONFIG$published_after
}

# Collect videos
all_videos <- list()
video_counter <- 1

# Strategy 1: Search by keywords
if (CONFIG$search_strategy %in% c("keywords", "both")) {
  log_message("Searching by keywords...")
  
  for (keyword in CONFIG$search_keywords) {
    log_message("  Keyword: ", keyword)
    videos <- search_videos_by_keyword(keyword, 
                                       CONFIG$max_results_per_query,
                                       published_after)
    
    if (nrow(videos) > 0) {
      all_videos[[video_counter]] <- videos
      video_counter <- video_counter + 1
    }
    
    Sys.sleep(CONFIG$api_delay_seconds)
  }
}

# Strategy 2: Search by channels
if (CONFIG$search_strategy %in% c("channels", "both")) {
  log_message("Fetching videos from channels...")
  
  for (channel_id in CONFIG$channel_ids) {
    log_message("  Channel: ", channel_id)
    videos <- get_channel_videos(channel_id,
                                 CONFIG$videos_per_channel,
                                 published_after)
    
    if (nrow(videos) > 0) {
      all_videos[[video_counter]] <- videos
      video_counter <- video_counter + 1
    }
    
    Sys.sleep(CONFIG$api_delay_seconds)
  }
}

# Combine and deduplicate videos
youtube_df <- bind_rows(all_videos) %>%
  distinct(video_id, .keep_all = TRUE)

log_message("Total unique videos found: ", nrow(youtube_df))

if (nrow(youtube_df) == 0) {
  log_message("No videos found. Exiting.", level = "ERROR")
  quit(save = "no", status = 1)
}

# Get video statistics
log_message("Fetching video statistics...")
video_stats <- get_video_statistics(youtube_df$video_id)

youtube_df <- youtube_df %>%
  left_join(video_stats, by = "video_id") %>%
  mutate(
    published_at = as.POSIXct(published_at, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    engagement_rate = (like_count + comment_count) / pmax(view_count, 1)
  )

# Clean and filter videos
youtube_df2 <- youtube_df %>%
  mutate(
    full_text = paste(title, description),
    clean_text = purrr::map_chr(full_text, clean_text),
    word_count = str_count(clean_text, "\\w+"),
    is_skin = purrr::map_lgl(clean_text, contains_skin_keyword)
  ) %>%
  filter(
    is_skin,
    nchar(clean_text) >= CONFIG$min_text_length,
    word_count >= CONFIG$min_word_count,
    view_count >= CONFIG$min_view_count,
    engagement_rate >= CONFIG$min_engagement_rate
  )

log_message("After filtering: ", nrow(youtube_df2), " videos remain")

# ========== FETCH COMMENTS ==========
all_comments <- tibble()

if (CONFIG$fetch_comments && nrow(youtube_df2) > 0) {
  log_message("Fetching comments from ", nrow(youtube_df2), " videos...")
  
  for (i in seq_len(min(nrow(youtube_df2), 100))) {  # Limit to avoid quota issues
    video_id <- youtube_df2$video_id[i]
    comments <- get_video_comments(video_id, CONFIG$max_comments_per_video)
    
    if (nrow(comments) > 0) {
      all_comments <- bind_rows(all_comments, comments)
    }
    
    if (i %% 10 == 0) {
      log_message("  Processed ", i, "/", nrow(youtube_df2), " videos")
    }
    
    Sys.sleep(CONFIG$api_delay_seconds)
  }
  
  log_message("Total comments fetched: ", nrow(all_comments))
  
  # Clean comments
  if (nrow(all_comments) > 0) {
    all_comments <- all_comments %>%
      mutate(
        published_at = as.POSIXct(published_at, format = "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
        clean_text = purrr::map_chr(comment_text, clean_text),
        word_count = str_count(clean_text, "\\w+"),
        is_skin = purrr::map_lgl(clean_text, contains_skin_keyword)
      ) %>%
      filter(is_skin, word_count >= CONFIG$min_word_count)
    
    log_message("Skin-related comments: ", nrow(all_comments))
  }
}

# ========== SENTIMENT ANALYSIS ==========
if (CONFIG$calculate_sentiment) {
  log_message("Calculating sentiment scores...")
  
  # Video sentiment (from titles/descriptions)
  video_sentiment <- youtube_df2 %>%
    select(video_id, clean_text) %>%
    unnest_tokens(word, clean_text) %>%
    inner_join(get_sentiments("bing"), by = "word") %>%
    count(video_id, sentiment) %>%
    pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>%
    mutate(sentiment_score = positive - negative)
  
  youtube_df2 <- youtube_df2 %>%
    left_join(video_sentiment, by = "video_id") %>%
    mutate(
      positive = replace_na(positive, 0),
      negative = replace_na(negative, 0),
      sentiment_score = replace_na(sentiment_score, 0)
    )
  
  # Comment sentiment
  if (nrow(all_comments) > 0) {
    comment_sentiment <- all_comments %>%
      select(comment_id, clean_text) %>%
      unnest_tokens(word, clean_text) %>%
      inner_join(get_sentiments("bing"), by = "word") %>%
      count(comment_id, sentiment) %>%
      pivot_wider(names_from = sentiment, values_from = n, values_fill = 0) %>%
      mutate(sentiment_score = positive - negative)
    
    all_comments <- all_comments %>%
      left_join(comment_sentiment, by = "comment_id") %>%
      mutate(
        positive = replace_na(positive, 0),
        negative = replace_na(negative, 0),
        sentiment_score = replace_na(sentiment_score, 0)
      )
  }
}

# ========== N-GRAM ANALYSIS ==========
log_message("Performing n-gram analysis...")

# Combine video and comment text
text_for_analysis <- bind_rows(
  youtube_df2 %>% select(clean_text) %>% mutate(source = "video"),
  all_comments %>% select(clean_text) %>% mutate(source = "comment")
)

ngram_results <- list()

# Unigrams
if (CONFIG$include_unigrams) {
  log_message("  Generating unigrams...")
  unigrams <- text_for_analysis %>%
    unnest_tokens(word, clean_text, token = "words") %>%
    anti_join(all_stopwords, by = "word") %>%
    count(word, source, sort = TRUE, name = "count") %>%
    group_by(word) %>%
    summarise(count = sum(count), .groups = "drop") %>%
    mutate(type = "unigram") %>%
    rename(ngram = word)
  
  ngram_results[["unigrams"]] <- unigrams
}

# Bigrams
if (CONFIG$include_bigrams) {
  log_message("  Generating bigrams...")
  bigrams <- text_for_analysis %>%
    unnest_tokens(bigram, clean_text, token = "ngrams", n = 2) %>%
    count(bigram, sort = TRUE, name = "count") %>%
    separate(bigram, into = c("w1", "w2"), sep = " ", remove = FALSE) %>%
    anti_join(all_stopwords, by = c("w1" = "word")) %>%
    anti_join(all_stopwords, by = c("w2" = "word")) %>%
    select(bigram, count) %>%
    mutate(type = "bigram") %>%
    rename(ngram = bigram)
  
  ngram_results[["bigrams"]] <- bigrams
}

# Trigrams
if (CONFIG$include_trigrams) {
  log_message("  Generating trigrams...")
  trigrams <- text_for_analysis %>%
    unnest_tokens(trigram, clean_text, token = "ngrams", n = 3) %>%
    count(trigram, sort = TRUE, name = "count") %>%
    separate(trigram, into = c("w1", "w2", "w3"), sep = " ", remove = FALSE) %>%
    anti_join(all_stopwords, by = c("w1" = "word")) %>%
    anti_join(all_stopwords, by = c("w2" = "word")) %>%
    anti_join(all_stopwords, by = c("w3" = "word")) %>%
    select(trigram, count) %>%
    mutate(type = "trigram") %>%
    rename(ngram = trigram)
  
  ngram_results[["trigrams"]] <- trigrams
}

# Combine n-grams
ngrams_combined <- bind_rows(ngram_results) %>%
  group_by(type) %>%
  mutate(
    percent = round((count / sum(count)) * 100, 2),
    rank = row_number()
  ) %>%
  ungroup() %>%
  arrange(type, desc(count))

ngrams_top <- ngrams_combined %>%
  group_by(type) %>%
  slice_head(n = CONFIG$top_k_results) %>%
  ungroup()

# ========== CHANNEL ANALYSIS ==========
if (CONFIG$analyze_by_channel) {
  log_message("Generating channel-specific analysis...")
  
  channel_stats <- youtube_df2 %>%
    group_by(channel_id, channel_title) %>%
    summarise(
      video_count = n(),
      total_views = sum(view_count, na.rm = TRUE),
      total_likes = sum(like_count, na.rm = TRUE),
      total_comments = sum(comment_count, na.rm = TRUE),
      avg_views = mean(view_count, na.rm = TRUE),
      avg_engagement_rate = mean(engagement_rate, na.rm = TRUE),
      avg_sentiment = if (CONFIG$calculate_sentiment) 
        mean(sentiment_score, na.rm = TRUE) else NA,
      .groups = "drop"
    ) %>%
    arrange(desc(total_views))
}

# ========== SUMMARY STATISTICS ==========
summary_stats <- list(
  analysis_timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  search_strategy = CONFIG$search_strategy,
  total_videos_found = nrow(youtube_df),
  videos_after_filtering = nrow(youtube_df2),
  total_comments = nrow(all_comments),
  channels_analyzed = n_distinct(youtube_df2$channel_id),
  date_range = if(nrow(youtube_df2) > 0) 
    paste(min(youtube_df2$published_at), "to", max(youtube_df2$published_at)) else "N/A",
  total_views = sum(youtube_df2$view_count, na.rm = TRUE),
  total_likes = sum(youtube_df2$like_count, na.rm = TRUE),
  avg_engagement_rate = mean(youtube_df2$engagement_rate, na.rm = TRUE),
  unique_unigrams = if (CONFIG$include_unigrams) nrow(ngram_results$unigrams) else 0,
  unique_bigrams = if (CONFIG$include_bigrams) nrow(ngram_results$bigrams) else 0,
  unique_trigrams = if (CONFIG$include_trigrams) nrow(ngram_results$trigrams) else 0
)

# ========== OUTPUT ==========
log_message("=== ANALYSIS COMPLETE ===")
log_message("Videos analyzed: ", summary_stats$videos_after_filtering)
log_message("Comments analyzed: ", summary_stats$total_comments)
log_message("Total views: ", format(summary_stats$total_views, big.mark = ","))
log_message("Unique terms: unigrams=", summary_stats$unique_unigrams,
            ", bigrams=", summary_stats$unique_bigrams,
            ", trigrams=", summary_stats$unique_trigrams)

# Display results
cat("\n=== TOP TERMS BY TYPE ===\n")
print(ngrams_top %>% select(type, ngram, count, percent, rank) %>% head(30))

if (CONFIG$analyze_by_channel) {
  cat("\n=== CHANNEL STATISTICS ===\n")
  print(channel_stats)
}

# Save outputs
if (CONFIG$save_outputs) {
  log_message("Saving outputs to: ", CONFIG$output_dir)
  
  if (CONFIG$save_video_metadata) {
    filename <- file.path(CONFIG$output_dir, 
                          paste0(CONFIG$output_prefix, "_videos.csv"))
    write_csv(youtube_df2, filename)
    log_message("Saved video metadata: ", filename)
  }
  
  if (CONFIG$save_comments && nrow(all_comments) > 0) {
    filename <- file.path(CONFIG$output_dir, 
                          paste0(CONFIG$output_prefix, "_comments.csv"))
    write_csv(all_comments, filename)
    log_message("Saved comments: ", filename)
  }
  
  if (CONFIG$save_ngrams) {
    filename <- file.path(CONFIG$output_dir, 
                          paste0(CONFIG$output_prefix, "_top_ngrams.csv"))
    write_csv(ngrams_top, filename)
    log_message("Saved top n-grams: ", filename)
    
    filename_full <- file.path(CONFIG$output_dir, 
                               paste0(CONFIG$output_prefix, "_all_ngrams.csv"))
    write_csv(ngrams_combined, filename_full)
    log_message("Saved all n-grams: ", filename_full)
  }
  
  if (CONFIG$save_summary_stats) {
    filename <- file.path(CONFIG$output_dir, 
                          paste0(CONFIG$output_prefix, "_summary.json"))
    write_json(summary_stats, filename, pretty = TRUE, auto_unbox = TRUE)
    log_message("Saved summary stats: ", filename)
    
    if (CONFIG$analyze_by_channel) {
      filename <- file.path(CONFIG$output_dir, 
                            paste0(CONFIG$output_prefix, "_channel_stats.csv"))
      write_csv(channel_stats, filename)
      log_message("Saved channel stats: ", filename)
    }
  }
}

log_message("YouTube analysis complete!")