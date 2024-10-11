user_data = [
    {
        "user_id": "12345",
        "name": "Aslon",
        "phone": "+777 10 10",
        "email": "aslon.hamidov@example.com",
        "debt_info": {
            "total_debt": 150000.00,
            "currency": "SUMM",
            "loan_type": "Personal Loan",
            "interest_rate": 5.75,
            "monthly_payment": 50000.00,
            "next_payment_due_date": "2024-10-01",
            "overdue_payments": 2,
            "last_payment_date": "2024-08-01",
            "last_payment_amount": 500.00,
            "penalties": {"late_fee": 50.00, "total_penalties": 100.00},
            "remaining_balance": 14500.00,
            "loan_start_date": "2022-01-01",
            "loan_due_date": "2027-01-01",
        },
        "status": "Delinquent",
        "contact_history": [
            {"date": "2024-08-15", "method": "Phone", "result": "No Answer"},
            {"date": "2024-07-25", "method": "Email", "result": "Reminder Sent"},
        ],
        "bank_details": {
            "bank_name": "BRB bank",
            "branch_name": "Main Branch",
            "branch_code": "001",
            "contact_number": "+888 55 44",
        },
    },
    {
        "user_id": "67890",
        "name": "Bekzod",
        "phone": "+998 90 123 45 67",
        "email": "bekzod.karimov@example.com",
        "debt_info": {
            "total_debt": 200000.00,
            "currency": "SUMM",
            "loan_type": "Car Loan",
            "interest_rate": 6.25,
            "monthly_payment": 60000.00,
            "next_payment_due_date": "2024-11-05",
            "overdue_payments": 1,
            "last_payment_date": "2024-09-05",
            "last_payment_amount": 600.00,
            "penalties": {"late_fee": 75.00, "total_penalties": 75.00},
            "remaining_balance": 19000.00,
            "loan_start_date": "2022-06-01",
            "loan_due_date": "2026-06-01",
        },
        "status": "Delinquent",
        "contact_history": [
            {"date": "2024-09-10", "method": "Phone", "result": "Left Message"},
            {
                "date": "2024-08-10",
                "method": "Email",
                "result": "Payment Reminder Sent",
            },
        ],
        "bank_details": {
            "bank_name": "BRB bank",
            "branch_name": "Main Branch",
            "branch_code": "001",
            "contact_number": "+888 55 44",
        },
    },
]
mdx_user_data_history = [
    {
        "patient_info": {
            "patient_id": "P123456",
            "first_name": "Иван",
            "last_name": "Иванов",
            "date_of_birth": "1985-04-15",
            "gender": "Male",
            "contact_details": {
                "phone_number": "+1-555-123-4567",
                "email": "ivanov@example.com",
                "address": {
                    "street": "123 Улица Вязов",
                    "city": "Спрингфилд",
                    "state": "IL",
                    "zip_code": "62704",
                    "country": "США",
                },
            },
            "emergency_contact": {
                "name": "Анна Иванова",
                "relationship": "Жена",
                "phone_number": "+1-555-987-6543",
            },
        },
        "medical_history": {
            "allergies": [
                {"name": "Пенициллин", "reaction": "Сыпь", "severity": "Умеренная"},
                {"name": "Арахис", "reaction": "Анафилаксия", "severity": "Тяжелая"},
            ],
            "current_medications": [
                {
                    "name": "Лизиноприл",
                    "dosage": "10 мг",
                    "frequency": "Один раз в день",
                },
                {
                    "name": "Метформин",
                    "dosage": "500 мг",
                    "frequency": "Два раза в день",
                },
            ],
            "past_medical_conditions": [
                {
                    "condition": "Гипертония",
                    "diagnosed_date": "2015-06-20",
                    "status": "Контролируемая",
                },
                {
                    "condition": "Сахарный диабет 2 типа",
                    "diagnosed_date": "2018-09-15",
                    "status": "Текущая",
                },
            ],
            "surgical_history": [
                {
                    "procedure": "Аппендэктомия",
                    "date": "2005-11-10",
                    "outcome": "Успешно",
                }
            ],
            "family_history": {
                "conditions": [
                    {
                        "condition": "Сердечное заболевание",
                        "relation": "Отец",
                        "age_at_diagnosis": 55,
                    },
                    {
                        "condition": "Рак молочной железы",
                        "relation": "Мать",
                        "age_at_diagnosis": 60,
                    },
                ]
            },
            "social_history": {
                "smoking_status": "Бывший курильщик",
                "alcohol_use": "Редко",
                "exercise": "Умеренная, 3 раза в неделю",
            },
            "immunizations": [
                {"vaccine": "Грипп", "date": "2023-10-01"},
                {"vaccine": "Бустер COVID-19", "date": "2024-02-15"},
            ],
            "last_visit": {
                "date": "2024-09-05",
                "reason": "Плановый осмотр",
                "notes": "Пациент чувствует себя хорошо, рекомендовано продолжать текущий режим приема лекарств.",
            },
        },
    },
    {
        "patient_info": {
            "patient_id": "P789012",
            "first_name": "Екатерина",
            "last_name": "Смирнова",
            "date_of_birth": "1992-12-03",
            "gender": "Female",
            "contact_details": {
                "phone_number": "+1-555-234-5678",
                "email": "smirnova@example.com",
                "address": {
                    "street": "456 Улица Дубов",
                    "city": "Ривертон",
                    "state": "CA",
                    "zip_code": "90210",
                    "country": "США",
                },
            },
            "emergency_contact": {
                "name": "Алексей Смирнов",
                "relationship": "Брат",
                "phone_number": "+1-555-876-5432",
            },
        },
        "medical_history": {
            "allergies": [
                {"name": "Морепродукты", "reaction": "Крапивница", "severity": "Легкая"}
            ],
            "current_medications": [
                {
                    "name": "Аторвастатин",
                    "dosage": "20 мг",
                    "frequency": "Один раз в день",
                }
            ],
            "past_medical_conditions": [
                {
                    "condition": "Астма",
                    "diagnosed_date": "2000-03-22",
                    "status": "Контролируется ингалятором",
                },
                {
                    "condition": "Мигрень",
                    "diagnosed_date": "2015-05-18",
                    "status": "Случайная",
                },
            ],
            "surgical_history": [
                {
                    "procedure": "Тонзиллэктомия",
                    "date": "2008-08-14",
                    "outcome": "Успешно",
                }
            ],
            "family_history": {
                "conditions": [
                    {
                        "condition": "Гипертония",
                        "relation": "Мать",
                        "age_at_diagnosis": 50,
                    }
                ]
            },
            "social_history": {
                "smoking_status": "Никогда не курила",
                "alcohol_use": "Социальное употребление",
                "exercise": "Регулярная, 5 раз в неделю",
            },
            "immunizations": [{"vaccine": "ВПЧ", "date": "2010-07-10"}],
            "last_visit": {
                "date": "2024-08-20",
                "reason": "Контроль астмы",
                "notes": "Рассмотрено использование ингалятора; состояние стабильное.",
            },
        },
    },
    {
        "patient_info": {
            "patient_id": "P345678",
            "first_name": "Дмитрий",
            "last_name": "Петров",
            "date_of_birth": "1978-01-25",
            "gender": "Male",
            "contact_details": {
                "phone_number": "+1-555-345-6789",
                "email": "petrov@example.com",
                "address": {
                    "street": "789 Улица Соснов",
                    "city": "Лейквуд",
                    "state": "TX",
                    "zip_code": "77001",
                    "country": "США",
                },
            },
            "emergency_contact": {
                "name": "Мария Петрова",
                "relationship": "Жена",
                "phone_number": "+1-555-654-3210",
            },
        },
        "medical_history": {
            "allergies": [
                {
                    "name": "Латекс",
                    "reaction": "Контактный дерматит",
                    "severity": "Умеренная",
                }
            ],
            "current_medications": [
                {"name": "Омепразол", "dosage": "20 мг", "frequency": "Один раз в день"}
            ],
            "past_medical_conditions": [
                {
                    "condition": "ГЭРБ",
                    "diagnosed_date": "2010-12-11",
                    "status": "Контролируется лекарствами",
                },
                {
                    "condition": "Высокий холестерин",
                    "diagnosed_date": "2016-07-24",
                    "status": "Под контролем",
                },
            ],
            "surgical_history": [
                {
                    "procedure": "Артроскопия колена",
                    "date": "2019-04-10",
                    "outcome": "Идет восстановление, рекомендована физиотерапия",
                }
            ],
            "family_history": {
                "conditions": [
                    {
                        "condition": "Диабет",
                        "relation": "Мать",
                        "age_at_diagnosis": 45,
                    },
                    {
                        "condition": "Инсульт",
                        "relation": "Отец",
                        "age_at_diagnosis": 70,
                    },
                ]
            },
            "social_history": {
                "smoking_status": "Никогда не курил",
                "alcohol_use": "Умеренное",
                "exercise": "Легкая, 2 раза в неделю",
            },
            "immunizations": [{"vaccine": "Столбняк", "date": "2022-03-18"}],
            "last_visit": {
                "date": "2024-09-10",
                "reason": "Контроль ГЭРБ",
                "notes": "Пациент отмечает улучшение симптомов при текущем лечении.",
            },
        },
    },
]


def get_medx_user_data(id):
    if id > len(mdx_user_data_history) - 1:
        return mdx_user_data_history[0]
    return mdx_user_data_history[id]
