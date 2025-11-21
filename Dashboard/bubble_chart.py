import sys
import pandas as pd
from pathlib import Path

# ensure repository root is on sys.path so we can import Backend modules
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Try to import the analysis helper from Backend; may be None in limited environments
try:
    from Backend import analyse_data
except Exception:
    analyse_data = None
try:
    from Backend import improvised_rag
except Exception:
    improvised_rag = None
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QPushButton, QStackedWidget
)
from PyQt6.QtCore import QThread, pyqtSignal


class AnalysisWorker(QThread):
    """Background worker to run analyse_data + improvised_rag without blocking UI.

    Emits a single dict payload via the `finished` signal when done.
    """
    finished = pyqtSignal(dict)

    def __init__(self, ngram: str):
        super().__init__()
        self.ngram = ngram

    def run(self):
        payload = {}
        try:
            if analyse_data is None:
                payload['error'] = 'analyse_data module unavailable.'
                self.finished.emit(payload)
                return

            # Run graph analysis
            try:
                result = analyse_data.analyse_skin_concern(str(self.ngram))
            except Exception as e:
                payload['error'] = f'Error in analyse_data: {e}'
                self.finished.emit(payload)
                return

            graph_lines = result.get('graph_data') or []
            graph_text = "\n".join(graph_lines) if graph_lines else "No related graph data found."
            analysis_text = result.get('analysis') or ""

            payload['graph_text'] = graph_text
            payload['analysis_text'] = analysis_text

            # Now run RAG if available
            if improvised_rag is None:
                payload['rag_sections'] = {
                    'answer': 'RAG module unavailable.',
                    'suggestions': '',
                    'sources': ''
                }
                self.finished.emit(payload)
                return

            try:
                index_path = str(REPO_ROOT / 'eucerin_faiss.index')
                meta_path = str(REPO_ROOT / 'eucerin_metadata.jsonl')

                hits = improvised_rag.retrieve(index_path, meta_path, str(self.ngram), top_k=5)
                prompt = improvised_rag.build_prompt(str(self.ngram), hits, insights=analysis_text)
                rag_result = improvised_rag.call_chat_model(prompt)
                rag_text = str(rag_result)

                # Try to parse structured sections from the rag_text
                import re
                m = re.search(r"###\s*Answer\s*(.*?)###\s*Product Improvement Suggestions\s*(.*?)###\s*Sources\s*(.*)", rag_text, re.S | re.I)
                if m:
                    answer = m.group(1).strip()
                    suggestions = m.group(2).strip()
                    sources = m.group(3).strip()
                else:
                    # best-effort splits
                    parts = re.split(r"###\s*Product Improvement Suggestions|###\s*Sources", rag_text, flags=re.I)
                    answer = parts[0].strip() if parts else rag_text
                    suggestions = parts[1].strip() if len(parts) > 1 else ''
                    sources = parts[2].strip() if len(parts) > 2 else ''

                # Also attempt to extract a Marketing section if present (## or ### headings)
                marketing = ''
                mm = re.search(r"#{2,}\s*Marketing\s*(.*?)(?:\n#{2,}|\Z)", rag_text, re.S | re.I)
                if mm:
                    marketing = mm.group(1).strip()

                payload['rag_sections'] = {
                    'answer': answer,
                    'suggestions': suggestions,
                    'sources': sources,
                    'marketing': marketing
                }

            except Exception as e:
                payload['rag_sections'] = {
                    'answer': f'Error running RAG: {e}',
                    'suggestions': '',
                    'sources': ''
                }

            self.finished.emit(payload)

        except Exception as e:
            payload['error'] = f'Unhandled worker error: {e}'
            self.finished.emit(payload)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


# -------------------------------
# Page 1: Bubble Chart Page
# -------------------------------
class BubbleChartPage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setWindowTitle("N-gram Analysis Bubble Chart")
        self.resize(1000, 800)

        # Layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Label for selected n-gram
        self.selected_label = QLabel("Click on a bubble to select an n-gram")
        self.selected_label.setStyleSheet("""
            font-size: 12px;
            font-weight: bold;
            color: #1f77b4;
            padding: 15px;
            border: 2px solid #1f77b4;
            border-radius: 2px;
        """)
        layout.addWidget(self.selected_label)

        # Load data
        self.df = pd.read_csv('./skin_ngrams_analysis.csv')

        # Save x-axis (n-grams) to a text file
        try:
            ngrams_list = self.df['ngram'].tolist()
            with open('./ngrams_xaxis.txt', 'w', encoding='utf-8') as f:
                for ngram in ngrams_list:
                    f.write(str(ngram) + '\n')
        except Exception as e:
            print(f"Warning: Could not save n-grams to file: {e}")

        # Create figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Draw bubbles
        self.scat = self.ax.scatter(
            self.df['ngram'],
            self.df['count'],
            s=self.df['count'] * 4,
            c=self.df['count'],
            cmap='viridis',
            alpha=0.7,
            edgecolors='white'
        )
        self.ax.set_xlabel('N-gram')
        self.ax.set_ylabel('Count')
        self.ax.set_title('N-gram Analysis Bubble Chart')
        self.fig.tight_layout()
        self.canvas.draw()

        # Connect click event
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # NEXT BUTTON
        next_btn = QPushButton("Next")
        next_btn.setStyleSheet("padding: 10px; font-size: 14px;")
        next_btn.clicked.connect(self.go_next)
        layout.addWidget(next_btn)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Find nearest point
        distances = ((self.df['ngram'].map(str).apply(hash) - hash(str(event.xdata)))**2 +
                     (self.df['count'] - event.ydata)**2)
        idx = distances.idxmin()
        ngram = self.df.iloc[idx]['ngram']
        count = self.df.iloc[idx]['count']
        self.selected_label.setText(f"{ngram} (Count: {count})")
        # store selection on the page and on the main window so other pages can access it
        self.selected_ngram = ngram
        try:
            self.parent.selected_ngram = ngram
        except Exception:
            self.parent.selected_ngram = ngram

    def go_next(self):
        # pass selected n-gram to second page before switching
        selected = getattr(self.parent, 'selected_ngram', None)
        if selected is not None and hasattr(self.parent, 'page2'):
            try:
                self.parent.page2.update_concerns(selected)
            except Exception as e:
                print(f"Error updating concerns page: {e}")

        self.parent.setCurrentIndex(1)  # Switch to page index 1


class SecondPage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Make layout expand properly
        self.setMinimumHeight(400)

        # Title
        title = QLabel("Most Talked Concerns")
        title.setStyleSheet("font-size: 22px; font-weight: bold; padding: 10px;")
        title.setMinimumHeight(40)
        layout.addWidget(title)

        # Placeholder concerns text
        self.concerns_text = QLabel(
            "• Dry skin and flakiness\n"
            "• Acne / breakouts\n"
            "• Hyperpigmentation & dark spots\n"
            "• Redness and irritation\n"
            "• Sensitivity from harsh products\n"
        )
        self.concerns_text.setStyleSheet("font-size: 16px; padding: 10px;")
        self.concerns_text.setWordWrap(True)
        self.concerns_text.setMinimumHeight(140)
        layout.addWidget(self.concerns_text)

        # Status / loading label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size: 14px; padding: 4px; color: #555;")
        layout.addWidget(self.status_label)

        # (Moved) The "What Eucerin Can Do" section and detailed RAG output
        # have been moved to a separate page. Use Next to view product/action suggestions.

        # Add stretch so text stays visible
        layout.addStretch(1)

        # Back button
        back_btn = QPushButton("Back")
        back_btn.setStyleSheet("padding: 10px; font-size: 14px;")
        back_btn.clicked.connect(self.go_back)
        layout.addWidget(back_btn)

        # Next button to go to the "What Eucerin Can Do" page
        next_btn = QPushButton("Next: What Eucerin Can Do")
        next_btn.setStyleSheet("padding: 10px; font-size: 14px;")
        next_btn.clicked.connect(self.go_third)
        layout.addWidget(next_btn)

    def go_back(self):
        self.parent.setCurrentIndex(0)
        self.parent.adjustSize()

    def go_third(self):
        # Navigate to the third page (What Eucerin Can Do)
        try:
            self.parent.setCurrentIndex(2)
        except Exception:
            pass

    def update_concerns(self, ngram):
        """
        Update the concerns_text label using the analysis from Backend.analyse_data.
        """
        if not ngram:
            return
        # Start background analysis worker to avoid blocking the UI
        try:
            self.status_label.setText(f"Analyzing '{ngram}' ...")
        except Exception:
            pass

        # If backend modules are missing, short-circuit and show message
        if analyse_data is None:
            self.concerns_text.setText(f"Analysis module unavailable. Selected: {ngram}")
            self.status_label.setText("")
            return

        # Create and run worker thread
        worker = AnalysisWorker(ngram)
        worker.finished.connect(self._on_analysis_finished)
        worker.start()
        # keep a reference so Python doesn't GC the thread
        self._worker = worker

    def _on_analysis_finished(self, payload):
        """Handle results emitted from AnalysisWorker."""
        # Clear status
        try:
            self.status_label.setText("")
        except Exception:
            pass

        # payload expected keys: graph_text, analysis_text, rag_sections (dict), error (optional)
        if not payload:
            self.concerns_text.setText("No results returned.")
            return

        if payload.get('error'):
            self.concerns_text.setText(f"Error: {payload.get('error')}")
            return

        graph_text = payload.get('graph_text', '')
        analysis_text = payload.get('analysis_text', '')
        rag_sections = payload.get('rag_sections', {})

        final = f"Graph Data:\n{graph_text}\n\nAnalysis:\n{analysis_text}"
        self.concerns_text.setText(final)

        # Forward payload to ThirdPage (if present) so it can render the RAG results
        try:
            if hasattr(self.parent, 'page3') and hasattr(self.parent.page3, 'update_from_payload'):
                self.parent.page3.update_from_payload(payload)
        except Exception:
            pass


class ThirdPage(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

        layout = QVBoxLayout()
        self.setLayout(layout)

        title = QLabel("What Eucerin Can Do")
        title.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px;")
        title.setMinimumHeight(40)
        layout.addWidget(title)

        self.summary_label = QLabel("Summary:\n\n")
        self.summary_label.setStyleSheet("font-size: 16px; padding: 10px;")
        self.summary_label.setWordWrap(True)
        self.summary_label.setMinimumHeight(120)
        layout.addWidget(self.summary_label)

        self.suggestions_label = QLabel("Product Improvement Suggestions:\n\n")
        self.suggestions_label.setStyleSheet("font-size: 16px; padding: 10px;")
        self.suggestions_label.setWordWrap(True)
        self.suggestions_label.setMinimumHeight(160)
        layout.addWidget(self.suggestions_label)

        self.sources_label = QLabel("Sources:\n\n")
        self.sources_label.setStyleSheet("font-size: 14px; padding: 10px; color: #333;")
        self.sources_label.setWordWrap(True)
        self.sources_label.setMinimumHeight(80)
        layout.addWidget(self.sources_label)

        self.marketing_label = QLabel("Marketing:\n\n")
        self.marketing_label.setStyleSheet("font-size: 14px; padding: 10px; color: #333;")
        self.marketing_label.setWordWrap(True)
        self.marketing_label.setMinimumHeight(80)
        layout.addWidget(self.marketing_label)

        # Back button to return to concerns
        back_btn = QPushButton("Back to Concerns")
        back_btn.setStyleSheet("padding: 10px; font-size: 14px;")
        back_btn.clicked.connect(self.go_back)
        layout.addWidget(back_btn)

    def go_back(self):
        try:
            self.parent.setCurrentIndex(1)
        except Exception:
            pass

    def update_from_payload(self, payload: dict):
        # Update structured fields from worker payload
        if not payload:
            return
        rag_sections = payload.get('rag_sections', {})
        self.summary_label.setText("Summary:\n\n" + rag_sections.get('answer', ''))
        self.suggestions_label.setText("Product Improvement Suggestions:\n\n" + rag_sections.get('suggestions', ''))
        self.sources_label.setText("Sources:\n\n" + rag_sections.get('sources', ''))
        self.marketing_label.setText("Marketing:\n\n" + rag_sections.get('marketing', ''))


# -------------------------------
# Main Stacked Window
# -------------------------------
class MainWindow(QStackedWidget):
    def __init__(self):
        super().__init__()

        self.page1 = BubbleChartPage(self)
        self.page2 = SecondPage(self)
        self.page3 = ThirdPage(self)

        self.addWidget(self.page1)
        self.addWidget(self.page2)
        self.addWidget(self.page3)


# -------------------------------
# Run the App
# -------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1000, 800)
    window.show()
    sys.exit(app.exec())
