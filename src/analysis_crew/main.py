#!/usr/bin/env python
import sys
from analysis_crew.crew import AnalysisCrew


def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'file_path': 'src/analysis_crew/files/tesla.csv'
    }
    AnalysisCrew().crew().kickoff(inputs=inputs)