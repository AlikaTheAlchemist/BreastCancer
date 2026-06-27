import gradio as gr
import numpy as np
import joblib
import os
import json

# Словарь для перевода названий признаков на русский с пояснениями
FEATURE_INFO = {
    'worst perimeter': {
        'name': 'Макс. периметр',
        'tooltip': 'Максимальный периметр ядра среди всех клеток. Ключевой признак злокачественности.'
    },
    'worst area': {
        'name': 'Макс. площадь',
        'tooltip': 'Максимальная площадь ядра. Значительное увеличение — тревожный признак.'
    },
    'mean concavity': {
        'name': 'Средняя вогнутость',
        'tooltip': 'Степень вогнутости контура ядра. Высокое значение характерно для раковых клеток.'
    },
    'worst concavity': {
        'name': 'Макс. вогнутость',
        'tooltip': 'Максимальная вогнутость контура. Сильная вогнутость — признак агрессивных клеток.'
    },
    'worst concave points': {
        'name': 'Макс. число вогнутостей',
        'tooltip': 'Максимальное количество вогнутостей. Указывает на сложность и агрессивность опухоли.'
    },
    'radius error': {
        'name': 'Ошибка радиуса',
        'tooltip': 'Стандартная ошибка измерения радиуса. Показывает вариабельность размеров ядер.'
    },
    'worst smoothness': {
        'name': 'Макс. гладкость',
        'tooltip': 'Минимальная гладкость среди всех клеток. Неровная поверхность характерна для рака.'
    },
    'worst radius': {
        'name': 'Макс. радиус',
        'tooltip': 'Максимальный радиус ядра. Важный показатель агрессивности опухоли.'
    },
    'mean compactness': {
        'name': 'Средняя компактность',
        'tooltip': 'Отношение периметра к площади. Высокое значение означает деформированную форму ядра.'
    },
    'mean concave points': {
        'name': 'Среднее число вогнутостей',
        'tooltip': 'Среднее количество вогнутых участков на контуре ядра.'
    }
}

class BreastCancerPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.feature_ranges = {}
        self.stats = {}
        self.load_model()
        self.load_stats()
        self.define_ranges_from_stats()
    
    def load_model(self):
        try:
            self.model = joblib.load('models/best_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            
            with open('models/feature_names.txt', 'r', encoding='utf-8') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
                
        except FileNotFoundError:
            print("Ошибка: Модель не найдена. Запустите train_model.py")
            raise
    
    def load_stats(self):
        try:
            with open('models/feature_stats.json', 'r', encoding='utf-8') as f:
                self.stats = json.load(f)
        except FileNotFoundError:
            print("Ошибка: Статистика не найдена. Запустите train_model.py")
            raise
    
    def define_ranges_from_stats(self):
        for feature in self.feature_names:
            if feature in self.stats:
                self.feature_ranges[feature] = {
                    'min': self.stats[feature]['min'],
                    'max': self.stats[feature]['max'],
                    'default': self.stats[feature]['mean']
                }
            else:
                self.feature_ranges[feature] = {'min': 0.0, 'max': 100.0, 'default': 50.0}
    
    def get_benign_values(self):
        values = []
        for feature in self.feature_names:
            if feature in self.stats:
                val = self.stats[feature]['benign_mean']
                if feature in self.feature_ranges:
                    min_val = self.feature_ranges[feature]['min']
                    max_val = self.feature_ranges[feature]['max']
                    val = max(min_val, min(max_val, val))
                values.append(val)
            else:
                values.append(self.feature_ranges[feature]['default'])
        return values
    
    def get_malignant_values(self):
        values = []
        for feature in self.feature_names:
            if feature in self.stats:
                val = self.stats[feature]['malignant_mean']
                if feature in self.feature_ranges:
                    min_val = self.feature_ranges[feature]['min']
                    max_val = self.feature_ranges[feature]['max']
                    val = max(min_val, min(max_val, val))
                values.append(val)
            else:
                values.append(self.feature_ranges[feature]['default'])
        return values
    
    def get_random_benign_values(self):
        base_values = self.get_benign_values()
        random_values = []
        for i, feature in enumerate(self.feature_names):
            base_val = base_values[i]
            if feature in self.stats:
                std = self.stats[feature]['benign_std']
                val = np.random.normal(base_val, std * 0.7)
            else:
                val = base_val * np.random.uniform(0.8, 1.2)
            
            if feature in self.feature_ranges:
                min_val = self.feature_ranges[feature]['min']
                max_val = self.feature_ranges[feature]['max']
                val = max(min_val, min(max_val, val))
            random_values.append(val)
        return random_values
    
    def get_random_malignant_values(self):
        base_values = self.get_malignant_values()
        random_values = []
        for i, feature in enumerate(self.feature_names):
            base_val = base_values[i]
            if feature in self.stats:
                std = self.stats[feature]['malignant_std']
                val = np.random.normal(base_val, std * 0.7)
            else:
                val = base_val * np.random.uniform(0.8, 1.2)
            
            if feature in self.feature_ranges:
                min_val = self.feature_ranges[feature]['min']
                max_val = self.feature_ranges[feature]['max']
                val = max(min_val, min(max_val, val))
            random_values.append(val)
        return random_values
    
    def validate_value(self, value, feature_name):
        if feature_name in self.feature_ranges:
            min_val = self.feature_ranges[feature_name]['min']
            max_val = self.feature_ranges[feature_name]['max']
            
            if value < min_val:
                return min_val
            elif value > max_val:
                return max_val
        return value
    
    def get_mean_values(self):
        mean_values = []
        for feature in self.feature_names:
            if feature in self.feature_ranges:
                default_val = self.feature_ranges[feature]['default']
                min_val = self.feature_ranges[feature]['min']
                max_val = self.feature_ranges[feature]['max']
                mean_values.append(max(min_val, min(max_val, default_val)))
            else:
                mean_values.append(0.5)
        return mean_values
    
    def predict(self, *args):
        try:
            validated_args = []
            validation_status = []
            
            for i, value in enumerate(args):
                feature_name = self.feature_names[i]
                min_val = self.feature_ranges[feature_name]['min']
                max_val = self.feature_ranges[feature_name]['max']
                
                is_invalid = value < min_val or value > max_val
                validation_status.append(is_invalid)
                
                validated_value = self.validate_value(value, feature_name)
                validated_args.append(validated_value)
            
            input_data = np.array(validated_args).reshape(1, -1)
            input_scaled = self.scaler.transform(input_data)
            
            prediction = self.model.predict(input_scaled)[0]
            probability = self.model.predict_proba(input_scaled)[0]
            
            if prediction == 0:
                result = "ЗЛОКАЧЕСТВЕННАЯ"
                confidence = probability[0] * 100
            else:
                result = "ДОБРОКАЧЕСТВЕННАЯ"
                confidence = probability[1] * 100
            
            warning_msg = ""
            if any(validation_status):
                warning_msg = "\n\nВНИМАНИЕ: Некоторые значения были скорректированы до допустимых пределов."
            
            detailed_report = f"""
РЕЗУЛЬТАТ ДИАГНОСТИКИ

Диагноз: {result}
Уверенность: {confidence:.2f}%

Вероятности:
  Злокачественная: {probability[0]*100:.2f}%
  Доброкачественная: {probability[1]*100:.2f}%

"""
            return detailed_report, confidence
            
        except Exception as e:
            return f"Ошибка: {str(e)}", None

def get_feature_info(feature_name):
    """Возвращает информацию о признаке"""
    if feature_name in FEATURE_INFO:
        return FEATURE_INFO[feature_name]
    return {
        'name': feature_name.replace('_', ' ').title(),
        'tooltip': 'Характеристика клеточного ядра'
    }

def create_interface():
    try:
        predictor = BreastCancerPredictor()
    except Exception as e:
        print(f"Ошибка инициализации: {e}")
        return None
    
    feature_names = predictor.feature_names
    feature_ranges = predictor.feature_ranges
    
    custom_css = """
    <style>
        .invalid-slider .gradio-slider {
            border: 2px solid #ff0000 !important;
            box-shadow: 0 0 10px rgba(255, 0, 0, 0.3) !important;
        }
        
        .invalid-slider label {
            color: #ff0000 !important;
            font-weight: bold !important;
        }
        
        .invalid-slider .gradio-slider .slider-track {
            background: linear-gradient(to right, #ff6b6b, #ff0000) !important;
        }
        
        .invalid-slider .gradio-slider .slider-thumb {
            background: #ff0000 !important;
            border-color: #cc0000 !important;
        }
        
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .invalid-slider {
            animation: blink 0.5s ease 3;
        }
        
        .validation-info {
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 4px;
            padding: 10px;
            margin: 10px 0;
            color: #856404;
        }
        
        .validation-info.error {
            background: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }
        
        .validation-info.success {
            background: #d4edda;
            border-color: #28a745;
            color: #155724;
        }
        
        .preset-benign {
            background: #4ECDC4 !important;
            border-color: #3dbdb5 !important;
        }
        
        .preset-benign:hover {
            background: #3dbdb5 !important;
        }
        
        .preset-malignant {
            background: #FF6B6B !important;
            border-color: #e55a5a !important;
        }
        
        .preset-malignant:hover {
            background: #e55a5a !important;
        }
        
        input[type="number"] {
            max-width: 80px !important;
        }
        
        input[type="number"]::-webkit-inner-spin-button,
        input[type="number"]::-webkit-outer-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        
        input[type="number"] {
            -moz-appearance: textfield;
        }
        
        .top-features-note {
            background: #e8f4fd;
            border-left: 4px solid #2196F3;
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 4px;
        }
        
        /* Стили для подсказок */
        .slider-with-tooltip {
            position: relative;
        }
        
        .tooltip-icon {
            display: inline-block;
            width: 18px;
            height: 18px;
            background: #2196F3;
            color: white;
            border-radius: 50%;
            text-align: center;
            line-height: 18px;
            font-size: 12px;
            font-weight: bold;
            cursor: help;
            margin-left: 8px;
            flex-shrink: 0;
        }
        
        .tooltip-icon:hover {
            background: #1976D2;
        }
        
        .slider-label-container {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        
        .slider-label-container label {
            margin-bottom: 0 !important;
        }
    </style>
    """
    
    with gr.Blocks(title="Диагностика рака груди", theme=gr.themes.Soft(), head=custom_css) as interface:
        gr.Markdown(""" # Диагностика рака груди """)
        
        
        
        validation_info = gr.HTML(
            value="<div class='validation-info success'>✅ Все значения в допустимых пределах</div>",
            visible=False
        )
        
        
        with gr.Row():
            with gr.Column(scale=2):
                inputs = []
                
                for idx, feature in enumerate(feature_names):
                    range_info = feature_ranges[feature]
                    info = get_feature_info(feature)
                    
                    # Создаем HTML с иконкой подсказки
                    label_html = f"""
                    <div class="slider-label-container">
                        <label>{info['name']}</label>
                        <span class="tooltip-icon" title="{info['tooltip']}">?</span>
                    </div>
                    """
                    
                    with gr.Group(elem_classes="slider-group") as group:
                        # Используем Markdown для отображения HTML с подсказкой
                        gr.HTML(label_html)
                        
                        slider = gr.Slider(
                            minimum=float(range_info['min']),
                            maximum=float(range_info['max']),
                            step=0.01,
                            label="",  # Пустой label, так как используем HTML
                            value=float(range_info['default']),
                            interactive=True,
                        )
                        inputs.append(slider)
            
            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="Результат диагностики",
                    lines=14,
                    interactive=False,
                    placeholder="Введите данные и нажмите 'Диагностировать'"
                )
                
                confidence_bar = gr.Slider(
                    label="Уверенность модели (%)",
                    minimum=0,
                    maximum=100,
                    value=0,
                    interactive=False
                )
                
                with gr.Row():
                    predict_btn = gr.Button("Диагностировать", variant="primary", size="lg")
                
                gr.Markdown("**Быстрая загрузка значений:**")
                
                with gr.Row():
                    benign_btn = gr.Button("Доброкачественная", variant="secondary", size="sm", elem_classes="preset-benign")
                    malignant_btn = gr.Button("Злокачественная", variant="secondary", size="sm", elem_classes="preset-malignant")
                
                with gr.Row():
                    random_benign_btn = gr.Button("Случайная доброкачественная", variant="secondary", size="sm")
                    random_malignant_btn = gr.Button("Случайная злокачественная", variant="secondary", size="sm")
                
                gr.Markdown("**Дополнительно:**")
                
                with gr.Row():
                    mean_btn = gr.Button("Средние показатели", variant="secondary", size="sm")
                    random_btn = gr.Button("Полностью случайные", variant="secondary", size="sm")
        
        
        
        def check_validation(*args):
            invalid_count = 0
            for i, value in enumerate(args):
                feature_name = feature_names[i]
                min_val = feature_ranges[feature_name]['min']
                max_val = feature_ranges[feature_name]['max']
                
                if value < min_val or value > max_val:
                    invalid_count += 1
            
        
        for slider in inputs:
            slider.change(
                fn=check_validation,
                inputs=inputs,
                outputs=validation_info
            )
        
        predict_btn.click(
            fn=predictor.predict,
            inputs=inputs,
            outputs=[output_text, confidence_bar]
        ).then(
            fn=check_validation,
            inputs=inputs,
            outputs=validation_info
        )
        
        benign_btn.click(
            fn=predictor.get_benign_values,
            inputs=[],
            outputs=inputs
        ).then(
            fn=check_validation,
            inputs=inputs,
            outputs=validation_info
        )
        
        malignant_btn.click(
            fn=predictor.get_malignant_values,
            inputs=[],
            outputs=inputs
        ).then(
            fn=check_validation,
            inputs=inputs,
            outputs=validation_info
        )
        
        random_benign_btn.click(
            fn=predictor.get_random_benign_values,
            inputs=[],
            outputs=inputs
        ).then(
            fn=check_validation,
            inputs=inputs,
            outputs=validation_info
        )
        
        random_malignant_btn.click(
            fn=predictor.get_random_malignant_values,
            inputs=[],
            outputs=inputs
        ).then(
            fn=check_validation,
            inputs=inputs,
            outputs=validation_info
        )
        
        mean_btn.click(
            fn=predictor.get_mean_values,
            inputs=[],
            outputs=inputs
        ).then(
            fn=check_validation,
            inputs=inputs,
            outputs=validation_info
        )
        
        def fill_random():
            random_values = []
            for feature in feature_names:
                range_info = feature_ranges[feature]
                random_values.append(np.random.uniform(range_info['min'], range_info['max']))
            return random_values
        
        random_btn.click(
            fn=fill_random,
            inputs=[],
            outputs=inputs
        ).then(
            fn=check_validation,
            inputs=inputs,
            outputs=validation_info
        )
    
    return interface

def main():
    interface = create_interface()
    
    if interface:
        print("Приложение запущено")
        interface.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7860,
            debug=False
        )
    else:
        print("Ошибка запуска приложения")

if __name__ == "__main__":
    main()