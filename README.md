# Posture Risk Assessment

This repository contains code and resources for assessing the risk associated with various postures using machine learning techniques. The aim is to provide a tool that can analyze body posture from images or videos and classify the risk level based on predefined criteria.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Models](#models)

## Introduction

Workplace ergonomics is crucial for preventing musculoskeletal disorders. This project leverages computer vision to automatically assess the risk level of different postures, helping to identify and mitigate potential health risks.

## Features

- **Posture Detection**: Detects human postures from live streaming video.
- **Body joint angles estimation**: Estimation of shoulder and trunk angles.
- **Visualization**: Provides visual feedback on detected postures and risk levels over time.

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

## Models
The repository uses MediaPipe for detecting human body landmarks. 
