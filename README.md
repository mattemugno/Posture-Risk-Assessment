# Posture Risk Assessment

This repository contains code and resources for assessing the risk associated with various postures using machine learning techniques. The aim is to provide a tool that can analyze body posture from images or videos and classify the risk level based on predefined criteria.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Workplace ergonomics is crucial for preventing musculoskeletal disorders. This project leverages computer vision and machine learning to automatically assess the risk level of different postures, helping to identify and mitigate potential health risks.

## Features

- **Posture Detection**: Detects human postures from images or video frames.
- **Risk Classification**: Classifies postures into different risk levels (e.g., low, medium, high).
- **Visualization**: Provides visual feedback on detected postures and risk levels.
- **Configurable Models**: Supports different machine learning models for detection and classification.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/mattemugno/Posture-Risk-Assessment.git
   cd Posture-Risk-Assessment
2. Create and activate a virtual environment:
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install the required packages:
   ```sh
   pip install -r requirements.txt
