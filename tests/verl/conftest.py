"""Pytest configuration for verl tests.

This module provides fixtures for mock datasets and utilities for testing
verl evaluators without network access.
"""

from __future__ import annotations

import pytest
from datasets import Dataset


@pytest.fixture
def medqa_verl_dataset() -> Dataset:
    """Create mock MedQA dataset in verl format (prompt/ground_truth/metadata)."""
    return Dataset.from_dict(
        {
            "prompt": [
                [
                    {
                        "role": "user",
                        "content": "A 45-year-old man presents with chest pain. What is the diagnosis?\n\nA. MI\nB. PE\nC. GERD\nD. Costochondritis",
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "A patient has elevated blood pressure. What is the treatment?\n\nA. ACE inhibitor\nB. Beta blocker\nC. Diuretic\nD. CCB",
                    }
                ],
            ],
            "ground_truth": [
                {"answer": "A", "answer_text": "MI"},
                {"answer": "A", "answer_text": "ACE inhibitor"},
            ],
            "metadata": [
                {
                    "question": "A 45-year-old man presents with chest pain. What is the diagnosis?"
                },
                {
                    "question": "A patient has elevated blood pressure. What is the treatment?"
                },
            ],
        }
    )


@pytest.fixture
def med_mcqa_verl_dataset() -> Dataset:
    """Create mock MedMCQA dataset in verl format."""
    return Dataset.from_dict(
        {
            "prompt": [
                [
                    {
                        "role": "user",
                        "content": "Which of the following is an ACE inhibitor?\n\nA. Lisinopril\nB. Metoprolol\nC. Amlodipine\nD. Losartan",
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "What is the mechanism of action of aspirin?\n\nA. COX inhibition\nB. Beta blockade\nC. Calcium channel blockade\nD. ACE inhibition",
                    }
                ],
            ],
            "ground_truth": [
                {"answer": "A"},
                {"answer": "A"},
            ],
            "metadata": [
                {"question": "Which of the following is an ACE inhibitor?"},
                {"question": "What is the mechanism of action of aspirin?"},
            ],
        }
    )


@pytest.fixture
def medbullets_verl_dataset() -> Dataset:
    """Create mock MedBullets dataset in verl format."""
    return Dataset.from_dict(
        {
            "prompt": [
                [
                    {
                        "role": "user",
                        "content": "A 30-year-old woman presents with fatigue. Labs show low hemoglobin.\n\nA. Iron deficiency\nB. B12 deficiency\nC. Folate deficiency\nD. Anemia of chronic disease",
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "What is the most common cause of community-acquired pneumonia?\n\nA. S. pneumoniae\nB. H. influenzae\nC. M. pneumoniae\nD. S. aureus",
                    }
                ],
            ],
            "ground_truth": [
                {"answer": "A"},
                {"answer": "A"},
            ],
            "metadata": [
                {
                    "question": "A 30-year-old woman presents with fatigue. Labs show low hemoglobin.",
                    "num_options": 4,
                },
                {
                    "question": "What is the most common cause of community-acquired pneumonia?",
                    "num_options": 4,
                },
            ],
        }
    )


@pytest.fixture
def medcasereasoning_verl_dataset() -> Dataset:
    """Create mock MedCaseReasoning dataset in verl format."""
    return Dataset.from_dict(
        {
            "prompt": [
                [
                    {
                        "role": "user",
                        "content": "Case: A 45-year-old male presents with chest pain and shortness of breath.\n\nWhat is the most likely diagnosis?",
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "Case: A 30-year-old female with fever and joint pain for 2 weeks.\n\nWhat is the most likely diagnosis?",
                    }
                ],
            ],
            "ground_truth": [
                {"target": "Acute Myocardial Infarction"},
                {"target": "Systemic Lupus Erythematosus"},
            ],
            "metadata": [
                {
                    "case_prompt": "A 45-year-old male presents with chest pain and shortness of breath."
                },
                {
                    "case_prompt": "A 30-year-old female with fever and joint pain for 2 weeks."
                },
            ],
        }
    )


@pytest.fixture
def medxpertqa_verl_dataset() -> Dataset:
    """Create mock MedXpertQA dataset in verl format."""
    return Dataset.from_dict(
        {
            "prompt": [
                [
                    {
                        "role": "user",
                        "content": "What is the first-line treatment for Type 2 diabetes?\n\nA. Metformin\nB. Insulin\nC. Sulfonylurea\nD. DPP-4 inhibitor",
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "Which imaging modality is best for detecting PE?\n\nA. CT angiography\nB. V/Q scan\nC. Chest X-ray\nD. D-dimer",
                    }
                ],
            ],
            "ground_truth": [
                {"answer": "A"},
                {"answer": "A"},
            ],
            "metadata": [
                {
                    "question": "What is the first-line treatment for Type 2 diabetes?",
                    "question_type": "reasoning",
                },
                {
                    "question": "Which imaging modality is best for detecting PE?",
                    "question_type": "understanding",
                },
            ],
        }
    )


@pytest.fixture
def metamedqa_verl_dataset() -> Dataset:
    """Create mock MetaMedQA dataset in verl format."""
    return Dataset.from_dict(
        {
            "prompt": [
                [
                    {
                        "role": "user",
                        "content": "What is the treatment of choice for H. pylori?\n\nA. Triple therapy\nB. Monotherapy\nC. Dual therapy\nD. Quadruple therapy",
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "Which antibiotic is contraindicated in pregnancy?\n\nA. Tetracycline\nB. Penicillin\nC. Cephalosporin\nD. Erythromycin",
                    }
                ],
            ],
            "ground_truth": [
                {"answer": "Triple therapy"},
                {"answer": "Tetracycline"},
            ],
            "metadata": [
                {"question": "What is the treatment of choice for H. pylori?"},
                {"question": "Which antibiotic is contraindicated in pregnancy?"},
            ],
        }
    )


@pytest.fixture
def mmlu_pro_health_verl_dataset() -> Dataset:
    """Create mock MMLUProHealth dataset in verl format."""
    return Dataset.from_dict(
        {
            "prompt": [
                [
                    {
                        "role": "user",
                        "content": "What is the mechanism of action of statins?\n\nA. HMG-CoA reductase inhibition\nB. Cholesterol absorption inhibition\nC. Bile acid sequestration\nD. PCSK9 inhibition",
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "Which vitamin deficiency causes scurvy?\n\nA. Vitamin C\nB. Vitamin D\nC. Vitamin B12\nD. Vitamin K",
                    }
                ],
            ],
            "ground_truth": [
                {"answer": "A"},
                {"answer": "A"},
            ],
            "metadata": [
                {
                    "question": "What is the mechanism of action of statins?",
                    "category": "health",
                },
                {
                    "question": "Which vitamin deficiency causes scurvy?",
                    "category": "health",
                },
            ],
        }
    )


@pytest.fixture
def pubhealthbench_verl_dataset() -> Dataset:
    """Create mock PubHealthBench dataset in verl format."""
    return Dataset.from_dict(
        {
            "prompt": [
                [
                    {
                        "role": "user",
                        "content": "Does smoking cause lung cancer?\n\nA. Yes\nB. No\nC. Maybe",
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "What is the recommended daily water intake?",
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "Is aspirin safe for children?\n\nA. Safe\nB. Not recommended\nC. Depends on age",
                    }
                ],
            ],
            "ground_truth": [
                {"answer": "A"},
                {"answer": "8 glasses per day"},
                {"answer": "B"},
            ],
            "metadata": [
                {"question": "Does smoking cause lung cancer?", "is_mcq": True},
                {
                    "question": "What is the recommended daily water intake?",
                    "is_mcq": False,
                },
                {"question": "Is aspirin safe for children?", "is_mcq": True},
            ],
        }
    )


@pytest.fixture
def pubmedqa_verl_dataset() -> Dataset:
    """Create mock PubMedQA dataset in verl format."""
    return Dataset.from_dict(
        {
            "prompt": [
                [
                    {
                        "role": "user",
                        "content": "Context: Diabetes is common. We studied 1000 patients.\n\nQuestion: Does metformin reduce mortality in diabetic patients?\n\nA. yes\nB. no\nC. maybe",
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "Context: CVD is the leading cause of death. Aspirin showed mixed results.\n\nQuestion: Is aspirin effective for primary prevention of CVD?\n\nA. yes\nB. no\nC. maybe",
                    }
                ],
            ],
            "ground_truth": [
                {"answer": "yes"},
                {"answer": "maybe"},
            ],
            "metadata": [
                {"question": "Does metformin reduce mortality in diabetic patients?"},
                {"question": "Is aspirin effective for primary prevention of CVD?"},
            ],
        }
    )


@pytest.fixture
def healthbench_verl_dataset() -> Dataset:
    """Create mock HealthBench dataset in verl format."""
    return Dataset.from_dict(
        {
            "prompt": [
                [{"role": "user", "content": "What should I do for a headache?"}],
                [{"role": "user", "content": "How do I manage diabetes?"}],
            ],
            "ground_truth": [
                {
                    "criteria": [
                        "Mentions rest and hydration",
                        "Suggests OTC pain relief",
                    ],
                    "points_list": [1, 2],
                },
                {
                    "criteria": [
                        "Mentions diet management",
                        "Recommends consulting a doctor",
                    ],
                    "points_list": [2, 1],
                },
            ],
            "metadata": [
                {"prompt_id": "hb_001", "difficulty": "regular"},
                {"prompt_id": "hb_002", "difficulty": "regular"},
            ],
        }
    )
