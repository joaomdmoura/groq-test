from crewai_tools import BaseTool

class GenerateMarkdownReportTool(BaseTool):
    name: str = "GenerateMarkdownReportTool"
    description: str = "Generates a comprehensive markdown report from analysis and chart outputs."

    def _run(self, initial_data_path: str, advanced_data_path: str, charts_path: str, output_path: str) -> str:
        try:
            with open(initial_data_path, 'r') as file:
                initial_data = file.read()
            with open(advanced_data_path, 'r') as file:
                advanced_data = file.read()

            markdown_content = f"# Analysis Report\n\n## Initial Data Analysis\n\n```\n{initial_data}\n```\n\n"
            markdown_content += f"## Advanced Data Analysis\n\n```\n{advanced_data}\n```\n\n"
            markdown_content += f"## Charts\n\n![Charts]({charts_path})\n"

            with open(output_path, 'w') as file:
                file.write(markdown_content)

            return f"Markdown report generated at {output_path}"
        except Exception as e:
            return f"Error: {e}"
