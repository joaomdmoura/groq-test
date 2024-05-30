from crewai_tools import BaseTool
import pandas as pd

class CSVAnalysisTool(BaseTool):
    name: str = "CSVAnalysisTool"
    description: str = "Reads a CSV file and returns the head, column data types, and basic statistics."

    def _run(self, file_path: str) -> str:
        df = pd.read_csv(file_path)
        head = df.head().to_string()
        dtypes = df.dtypes.to_string()
        stats = df.describe().to_string()
        return f"Head of the CSV:\n{head}\n\nData Types:\n{dtypes}\n\nBasic Statistics:\n{stats}"