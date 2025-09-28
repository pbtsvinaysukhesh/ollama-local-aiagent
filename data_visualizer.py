# data_visualizer.py
import matplotlib.pyplot as plt
import os
import json

class DataVisualizer:
    def __init__(self, output_dir="output/charts"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

    def generate_chart(self, chart_type: str, chart_data_json: str, filename: str) -> str | None:
        try: chart_data = json.loads(chart_data_json)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON for chart data. {e}"); return None
        plt.style.use('seaborn-v0_8-talk'); fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        try:
            if chart_type.lower() == 'bar':
                labels = list(chart_data['data'].keys()); values = list(chart_data['data'].values())
                ax.bar(labels, values); ax.set_ylabel(chart_data.get('y_label', 'Values'))
                ax.tick_params(axis='x', rotation=45)
            elif chart_type.lower() == 'pie':
                labels = list(chart_data['data'].keys()); values = list(chart_data['data'].values())
                ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90); ax.axis('equal')
            else: print(f"Error: Unsupported chart type '{chart_type}'"); return None
            ax.set_title(chart_data.get('title', 'Untitled Chart'), pad=20)
            ax.set_xlabel(chart_data.get('x_label', ''))
            fig.tight_layout(); filepath = os.path.join(self.output_dir, f"{filename}.png")
            fig.savefig(filepath); plt.close(fig); print(f"Chart saved to {filepath}")
            return filepath
        except Exception as e:
            print(f"An error during chart generation: {e}"); plt.close(fig); return None