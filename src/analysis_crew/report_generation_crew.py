from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

from langchain_groq import ChatGroq

@CrewBase
class ReportGeneratorCrew:
    """Report Generator crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    search_tool = SerperDevTool()
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

    @agent
    def content_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['content_writer'],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    @task
    def write_markdown_report(self) -> Task:
        return Task(
            config=self.tasks_config['write_markdown_report'],
            agent=self.content_writer()
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