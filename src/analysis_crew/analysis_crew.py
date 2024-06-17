from typing import List, Optional
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field

from .tools.csv_analysis_tool import CSVAnalysisTool
from .tools.code_execution_tool import CodeExecutionTool

from langchain_groq import ChatGroq

class Charts(BaseModel):
    charts_images: List[str] = Field(..., title="list of charts images paths")

class ReportSection(BaseModel):
    title: str = Field(..., title="title of the section")
    data: str = Field(..., title="data of the section")
    charts: List[str] = Field(..., title="list of chart images paths")
    long_draft: str = Field(..., title="long draft of the section")
    why_it_matters: str = Field(..., title="why it matters")
    references: List[str] = Field(..., title="URL and information about the references for the content")

class Report(BaseModel):
    sections: List[ReportSection] = Field(..., title="list of sections")

@CrewBase
class AnalysisCrew:
    """Analysis crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    csv_tool = CSVAnalysisTool()
    code_tool = CodeExecutionTool()
    search_tool = SerperDevTool()
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

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
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[self.search_tool],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    @agent
    def content_planner(self) -> Agent:
        return Agent(
            config=self.agents_config['content_planner'],
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
    def research_company(self) -> Task:
        return Task(
            config=self.tasks_config['research_company'],
            tools=[self.search_tool],
            agent=self.researcher()
        )

    @task
    def plan_report(self) -> Task:
        return Task(
            config=self.tasks_config['plan_report'],
            context=[
                self.initial_data_analysis(),
                self.advanced_data_analysis(),
                self.generate_charts(),
                self.research_company()
            ],
            agent=self.content_planner(),
            output_pydantic=Report
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