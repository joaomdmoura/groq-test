from crewai_tools import BaseTool

class CodeExecutionTool(BaseTool):
    name: str = "CodeExecutionTool"
    description: str = "Executes a given string of Python code using exec and returns the result."

    def _run(self, code: str) -> str:
        try:
            # Use a limited dictionary for locals() to avoid unwanted access
            local_vars = {}
            exec(code, {}, local_vars)
            return str(local_vars)
        except Exception as e:
            return f"Error: {e}"
