from typing import List, Optional
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field

from .tools.csv_analysis_tool import CSVAnalysisTool
from .tools.code_execution_tool import CodeExecutionTool

from langchain_groq import ChatGroq

class Charts(BaseModel):
    charts_images: List[str] = Field(..., title="Charts")

@CrewBase
class AnalysisCrew:
    """Analysis crew"""
    agents_config_path = 'config/agents.yaml'
    tasks_config_path = 'config/tasks.yaml'
    csv_tool = CSVAnalysisTool()
    code_tool = CodeExecutionTool()
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
    # llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['data_analyst'],
            tools=[self.csv_tool, self.code_tool],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    @agent
    def data_scientist(self) -> Agent:
        return Agent(
            config=self.agents_config['data_scientist'],
            tools=[self.csv_tool, self.code_tool],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    @agent
    def content_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['content_writer'],
            verbose=True,
            allow_delegation=False,
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
            agent=self.data_analyst(),
            output_pydantic=Charts
        )

    @task
    def generate_markdown_report(self) -> Task:
        return Task(
            config=self.tasks_config['generate_report_plan'],
            context=[self.initial_data_analysis(), self.advanced_data_analysis(), self.generate_charts()],
            agent=self.data_analyst()
        )

    @task
    def write_markdown_report(self) -> Task:
        return Task(
            config=self.tasks_config['write_markdown_report'],
            agent=self.content_writer(),
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AnalysisCrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=2,
        )