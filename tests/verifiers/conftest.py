"""Pytest configuration for verifiers tests.

This module provides fixtures for mock datasets and utilities for testing
verifiers evaluators without network access.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from datasets import Dataset


@pytest.fixture
def medqa_mock_dataset() -> Dataset:
    """Create mock MedQA dataset with representative rows."""
    return Dataset.from_dict(
        {
            "question": [
                "A 45-year-old man presents with chest pain. What is the diagnosis?",
                "A patient has elevated blood pressure. What is the treatment?",
            ],
            "options": [
                {"A": "MI", "B": "PE", "C": "GERD", "D": "Costochondritis"},
                {
                    "A": "ACE inhibitor",
                    "B": "Beta blocker",
                    "C": "Diuretic",
                    "D": "CCB",
                },
            ],
            "answer_idx": ["A", "A"],
        }
    )


@pytest.fixture
def med_mcqa_mock_dataset() -> Dataset:
    """Create mock MedMCQA dataset with representative rows."""
    return Dataset.from_dict(
        {
            "question": [
                "Which of the following is an ACE inhibitor?",
                "What is the mechanism of action of aspirin?",
            ],
            "opa": ["Lisinopril", "COX inhibition"],
            "opb": ["Metoprolol", "Beta blockade"],
            "opc": ["Amlodipine", "Calcium channel blockade"],
            "opd": ["Losartan", "ACE inhibition"],
            "cop": [1, 1],
        }
    )


@pytest.fixture
def medbullets_mock_dataset() -> Dataset:
    """Create mock MedBullets dataset with representative rows."""
    return Dataset.from_dict(
        {
            "question": [
                "A 30-year-old woman presents with fatigue. Labs show low hemoglobin.",
                "What is the most common cause of community-acquired pneumonia?",
            ],
            "options": [
                {
                    "A": "Iron deficiency",
                    "B": "B12 deficiency",
                    "C": "Folate deficiency",
                    "D": "Anemia of chronic disease",
                },
                {
                    "A": "S. pneumoniae",
                    "B": "H. influenzae",
                    "C": "M. pneumoniae",
                    "D": "S. aureus",
                },
            ],
            "answer": ["A", "A"],
        }
    )


@pytest.fixture
def medxpertqa_mock_dataset() -> Dataset:
    """Create mock MedXpertQA dataset with representative rows."""
    return Dataset.from_dict(
        {
            "question": [
                "What is the first-line treatment for Type 2 diabetes?",
                "Which imaging modality is best for detecting PE?",
            ],
            "options": [
                {
                    "A": "Metformin",
                    "B": "Insulin",
                    "C": "Sulfonylurea",
                    "D": "DPP-4 inhibitor",
                },
                {
                    "A": "CT angiography",
                    "B": "V/Q scan",
                    "C": "Chest X-ray",
                    "D": "D-dimer",
                },
            ],
            "label": ["A", "A"],
            "question_type": ["reasoning", "understanding"],
        }
    )


@pytest.fixture
def metamedqa_mock_dataset() -> Dataset:
    """Create mock MetaMedQA dataset with representative rows."""
    return Dataset.from_dict(
        {
            "question": [
                "What is the treatment of choice for H. pylori?",
                "Which antibiotic is contraindicated in pregnancy?",
            ],
            "options": [
                {
                    "A": "Triple therapy",
                    "B": "Monotherapy",
                    "C": "Dual therapy",
                    "D": "Quadruple therapy",
                },
                {
                    "A": "Tetracycline",
                    "B": "Penicillin",
                    "C": "Cephalosporin",
                    "D": "Erythromycin",
                },
            ],
            "answer": ["Triple therapy", "Tetracycline"],
        }
    )


@pytest.fixture
def mmlu_pro_health_mock_dataset() -> Dataset:
    """Create mock MMLUProHealth dataset with representative rows."""
    return Dataset.from_dict(
        {
            "question": [
                "What is the mechanism of action of statins?",
                "Which vitamin deficiency causes scurvy?",
            ],
            "options": [
                [
                    "HMG-CoA reductase inhibition",
                    "Cholesterol absorption inhibition",
                    "Bile acid sequestration",
                    "PCSK9 inhibition",
                ],
                ["Vitamin C", "Vitamin D", "Vitamin B12", "Vitamin K"],
            ],
            "answer": ["A", "A"],
            "category": ["health", "health"],
        }
    )


@pytest.fixture
def pubmedqa_mock_dataset() -> Dataset:
    """Create mock PubMedQA dataset in openlifescienceai/pubmedqa schema."""
    return Dataset.from_list(
        [
            {
                "data": {
                    "Question": "Does metformin reduce mortality in diabetic patients?",
                    "Correct Option": "A",
                    "Context": ["Diabetes is common.", "We studied 1000 patients."],
                    "Options": {"A": "Yes", "B": "No", "C": "Maybe"},
                }
            },
            {
                "data": {
                    "Question": "Is aspirin effective for primary prevention of CVD?",
                    "Correct Option": "C",
                    "Context": [
                        "CVD is the leading cause of death.",
                        "Aspirin showed mixed results.",
                    ],
                    "Options": {"A": "Yes", "B": "No", "C": "Maybe"},
                }
            },
        ]
    )


@pytest.fixture
def pubhealthbench_mock_dataset() -> Dataset:
    """Create mock PubHealthBench dataset with representative rows."""
    return Dataset.from_dict(
        {
            "question": [
                "Does smoking cause lung cancer?",
                "What is the recommended daily water intake?",
                "Is aspirin safe for children?",
            ],
            "answer": [
                "A",
                "8 glasses per day",
                "B",
            ],
            "options": [
                ["Yes", "No", "Maybe"],
                [],
                ["Safe", "Not recommended", "Depends on age"],
            ],
            "source": ["study1", "guideline2", "study3"],
        }
    )


@pytest.fixture
def medcasereasoning_mock_dataset() -> Dataset:
    """Create mock MedCaseReasoning dataset with representative rows."""
    return Dataset.from_dict(
        {
            "case_prompt": [
                "A 45-year-old male presents with chest pain and shortness of breath.",
                "A 30-year-old female with fever and joint pain for 2 weeks.",
            ],
            "final_diagnosis": [
                "Acute Myocardial Infarction",
                "Systemic Lupus Erythematosus",
            ],
            "differential_diagnosis": [
                ["MI", "Angina", "PE"],
                ["SLE", "RA", "Viral arthritis"],
            ],
        }
    )


@pytest.fixture
def healthbench_mock_dataset() -> Dataset:
    """Create mock HealthBench dataset with representative rows."""
    return Dataset.from_dict(
        {
            "prompt": [
                [{"role": "user", "content": "What should I do for a headache?"}],
                [{"role": "user", "content": "How do I manage diabetes?"}],
            ],
            "prompt_id": ["hb_001", "hb_002"],
            "rubrics": [
                [
                    {
                        "criterion": "Mentions rest and hydration",
                        "points": 1,
                        "tags": ["axis:completeness"],
                    },
                    {
                        "criterion": "Suggests OTC pain relief",
                        "points": 2,
                        "tags": ["axis:accuracy"],
                    },
                ],
                [
                    {
                        "criterion": "Mentions diet management",
                        "points": 2,
                        "tags": ["axis:completeness"],
                    },
                    {
                        "criterion": "Recommends consulting a doctor",
                        "points": 1,
                        "tags": ["axis:safety"],
                    },
                ],
            ],
            "ideal_completions_data": [
                ["Rest, stay hydrated, and consider taking acetaminophen."],
                ["Monitor your diet and consult with your healthcare provider."],
            ],
        }
    )


@pytest.fixture
def mock_load_dataset_factory():
    """Factory fixture to create a mock load_dataset that returns a given dataset."""

    def _factory(dataset: Dataset) -> MagicMock:
        return MagicMock(return_value=dataset)

    return _factory
