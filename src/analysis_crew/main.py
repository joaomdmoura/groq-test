#!/usr/bin/env python
from analysis_crew.analysis_crew import AnalysisCrew
from analysis_crew.report_generation_crew import ReportGeneratorCrew


def run():
    inputs = {
        'file_path': 'src/analysis_crew/files/tesla.csv'
    }

    report = []
    report_plan = AnalysisCrew().crew().kickoff(inputs=inputs)

    print("Report Plan: ", report_plan.json())

    for section in report_plan.sections:
        plan = section.json()
        inputs = {'plan': plan}
        report_generator_crew = ReportGeneratorCrew()
        report_section = report_generator_crew.crew().kickoff(inputs=inputs)
        report.append(report_section)

    r = '\n\n'.join(report)
    with open('report.md', 'w') as f:
        f.write(r)

    print(r)


if __name__ == '__main__':
    run()
