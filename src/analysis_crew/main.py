#!/usr/bin/env python
import sys
from analysis_crew.crew import AnalysisCrewCrew


def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'file_path': 'src/analysis_crew/files/tesla.csv'
    }
    AnalysisCrewCrew().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    try:
        AnalysisCrewCrew().crew().train(n_iterations=int(sys.argv[1]))

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")
