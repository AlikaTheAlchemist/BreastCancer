import numpy as np
import pandas as pd
import joblib
import os
import json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def create_directories():
    dirs = ['models', 'plots', 'reports']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

def load_and_prepare_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y

def get_top_features(X, y, n_features=10):
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X, y)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_temp.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance.head(n_features)['feature'].tolist()
    return top_features, feature_importance

def train_model(X_train_scaled, X_test_scaled, y_train, y_test):
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    
    results = {
        'model': 'Random Forest',
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, pos_label=0),
        'recall': recall_score(y_test, y_pred, pos_label=0),
        'y_pred': y_pred
    }
    
    return rf, results

def get_statistics(X, y, top_features):
    stats = {}
    
    for feature in top_features:
        # Данные для доброкачественных (y=1)
        benign_data = X[y == 1][feature]
        # Данные для злокачественных (y=0)
        malignant_data = X[y == 0][feature]
        
        stats[feature] = {
            'min': float(X[feature].min()),
            'max': float(X[feature].max()),
            'mean': float(X[feature].mean()),
            'benign_mean': float(benign_data.mean()),
            'malignant_mean': float(malignant_data.mean()),
            'benign_std': float(benign_data.std()),
            'malignant_std': float(malignant_data.std()),
            'benign_min': float(benign_data.min()),
            'benign_max': float(benign_data.max()),
            'malignant_min': float(malignant_data.min()),
            'malignant_max': float(malignant_data.max())
        }
    
    return stats

def save_stats(stats, filename='models/feature_stats.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Статистика сохранена в {filename}")

def load_stats(filename='models/feature_stats.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_feature_importance(feature_importance, top_features):
    plt.figure(figsize=(12, 8))
    
    colors = ['#4ECDC4' if f in top_features else '#95a5a6' for f in feature_importance['feature']]
    
    plt.barh(feature_importance['feature'], feature_importance['importance'], color=colors)
    plt.xlabel('Важность')
    plt.title('Важность признаков (выделены топ-10)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_test, y_pred):
    """Визуализация матрицы ошибок"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Злокачественная (0)', 'Доброкачественная (1)'],
                yticklabels=['Злокачественная (0)', 'Доброкачественная (1)'])
    plt.title('Матрица ошибок - Random Forest (топ-10 признаков)')
    plt.ylabel('Фактический')
    plt.xlabel('Предсказанный')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_models(model, scaler, feature_names):
    joblib.dump(model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    with open('models/feature_names.txt', 'w', encoding='utf-8') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")

def save_report(y_test, y_pred, results):
    report = classification_report(y_test, y_pred, 
                                   target_names=['Злокачественная (0)', 'Доброкачественная (1)'])
    
    with open('reports/model_report.txt', 'w', encoding='utf-8') as f:
        f.write("ОТЧЕТ ОБ ОБУЧЕНИИ МОДЕЛИ\n")
        f.write(f"Модель: Random Forest\n")
        f.write(f"Точность: {results['accuracy']:.4f}\n")
        f.write(f"F1-Score (Злокачественные): {results['f1_score']:.4f}\n")
        f.write(f"Полнота (Злокачественные): {results['recall']:.4f}\n\n")
        f.write("Отчет по классификации:\n")
        f.write(report)

def print_statistics(stats):
    """Выводит статистику в консоль"""
    print("\nСтатистика по признакам:")
    print("="*60)
    for feature, values in stats.items():
        print(f"\n{feature}:")
        print(f"  Диапазон: {values['min']:.3f} - {values['max']:.3f}")
        print(f"  Среднее (доброкачественные): {values['benign_mean']:.3f}")
        print(f"  Среднее (злокачественные): {values['malignant_mean']:.3f}")
        print(f"  Разница: {values['malignant_mean'] - values['benign_mean']:.3f}")

def main():
    print("Обучение модели с отбором топ-10 признаков")
    
    create_directories()
    X, y = load_and_prepare_data()
    
    print(f"Размер датасета: {X.shape}")
    print(f"Злокачественные: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    print(f"Доброкачественные: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    
    # Определяем топ-10 признаков
    print("\nОпределение топ-10 важнейших признаков...")
    top_features, feature_importance = get_top_features(X, y, n_features=10)
    
    print("\nТоп-10 признаков:")
    for i, feature in enumerate(top_features, 1):
        importance = feature_importance[feature_importance['feature'] == feature]['importance'].values[0]
        print(f"{i:2d}. {feature}: {importance:.4f}")
    
    # Вычисляем статистику для топ-признаков
    stats = get_statistics(X, y, top_features)
    save_stats(stats)
    print_statistics(stats)
    
    # Оставляем только топ-10 признаков
    X_top = X[top_features]
    
    # Разделение на выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X_top, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nОбучающая выборка: {X_train.shape[0]} записей, {X_train.shape[1]} признаков")
    print(f"Тестовая выборка: {X_test.shape[0]} записей, {X_test.shape[1]} признаков")
    
    # Стандартизация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение модели
    model, results = train_model(X_train_scaled, X_test_scaled, y_train, y_test)
    
    print(f"\nРезультаты:")
    print(f"Точность: {results['accuracy']:.4f}")
    print(f"F1-Score (Злокачественные): {results['f1_score']:.4f}")
    
    print("\nОтчет по классификации:")
    print(classification_report(y_test, results['y_pred'], 
                                target_names=['Злокачественная', 'Доброкачественная']))
    
    # Сохранение
    save_models(model, scaler, top_features)
    
    # Визуализации
    plot_feature_importance(feature_importance, top_features)
    plot_confusion_matrix(y_test, results['y_pred'])
    save_report(y_test, results['y_pred'], results)
    
    print("\nОбучение завершено успешно")
    print("Сохраненные файлы:")
    print("  models/best_model.pkl")
    print("  models/scaler.pkl")
    print("  models/feature_names.txt")
    print("  models/feature_stats.json")
    print("  plots/feature_importance.png")
    print("  plots/confusion_matrix.png")
    print("  reports/model_report.txt")

if __name__ == "__main__":
    main()
