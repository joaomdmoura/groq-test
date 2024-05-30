from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from .tools.csv_analysis_tool import CSVAnalysisTool
from .tools.code_execution_tool import CodeExecutionTool
from .tools.generate_markdown_report_tool import GenerateMarkdownReportTool

from langchain_groq import ChatGroq

@CrewBase
class AnalysisCrewCrew:
    """AnalysisCrew crew"""
    agents_config_path = 'config/agents.yaml'
    tasks_config_path = 'config/tasks.yaml'
    csv_tool = CSVAnalysisTool()
    code_tool = CodeExecutionTool()
    markdown_tool = GenerateMarkdownReportTool()
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
    # llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['data_analyst'],
            tools=[self.csv_tool, self.code_tool, self.markdown_tool],
            verbose=True,
            llm=self.llm
        )

    @agent
    def data_scientist(self) -> Agent:
        return Agent(
            config=self.agents_config['data_scientist'],
            tools=[self.csv_tool, self.code_tool],
            verbose=True,
            llm=self.llm
        )

    @agent
    def qa_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['qa_analyst'],
            verbose=True,
            llm=self.llm
        )

    @task
    def initial_data_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['initial_data_analysis'],
            tools=[self.csv_tool, self.code_tool],
            agent=self.data_analyst()
        )

    @task
    def advanced_data_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['advanced_data_analysis'],
            tools=[self.csv_tool, self.code_tool],
            agent=self.data_scientist()
        )

    @task
    def generate_charts(self) -> Task:
        return Task(
            config=self.tasks_config['generate_charts'],
            tools=[self.csv_tool, self.code_tool],
            agent=self.data_analyst()
        )

    @task
    def code_review(self) -> Task:
        return Task(
            config=self.tasks_config['code_review'],
            agent=self.qa_analyst()
        )

    @task
    def generate_markdown_report(self) -> Task:
        return Task(
            config=self.tasks_config['generate_markdown_report'],
            tools=[self.markdown_tool],
            agent=self.data_analyst()
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AnalysisCrew crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=2,
        )
