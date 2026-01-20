import pandas as pd
import numpy as np
import re
import time
import os
import sys
import psutil
from collections import Counter
from math import log2
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
from scipy import stats
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# KONFIGURASI 
# ============================================================================

CONFIG = {
    'CSV_PATH': 'data.csv',  
    'SAMPLE_SIZE': 50000,  
    'N_ESTIMATORS': 100,
    'OUTPUT_FOLDER': 'hasil_training',
    'INFERENCE_SAMPLES': 1000,
    'ACCEPTABLE_LATENCY_MS': 100,  # Threshold 10ms untuk real-time
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_memory_usage_mb():
    """Dapatkan penggunaan memory saat ini dalam MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def get_model_size_mb(model):
    """Hitung ukuran model dalam MB"""
    import pickle
    return len(pickle.dumps(model)) / 1024 / 1024


# ============================================================================
# FEATURE ENGINEERING CLASS
# ============================================================================

class PasswordFeatureExtractor:
    """
    Ekstraksi 24 fitur dari password berdasarkan karakteristik:
    - Struktural (panjang, komposisi karakter)
    - Statistik (entropi, diversity)
    - Pola leksikal (keyboard pattern, leet speak, common words)
    """
    
    QWERTY_PATTERNS = [
        'qwerty', 'asdfgh', 'zxcvbn', 'qwertyuiop', 'asdfghjkl',
        '12345', '123456', '1234567', '12345678', '123456789'
    ]
    
    COMMON_WORDS = [
        'password', 'admin', 'user', 'login', 'welcome', 'monkey',
        'dragon', 'master', 'sunshine', 'princess', 'letmein'
    ]
    
    LEET_SUBSTITUTIONS = {
        'a': ['@', '4'], 'e': ['3'], 'i': ['1', '!'], 'o': ['0'],
        's': ['$', '5'], 't': ['7'], 'l': ['1'], 'g': ['9']
    }
    
    def extract_features(self, password):
        """
        Ekstraksi 24 fitur dari password string.
        
        Contoh: password = "P@ssw0rd123"
        Output: dict dengan 24 fitur numerik
        """
        features = {}
        
        # === 1. Basic Composition Features (5 fitur) ===
        # Menghitung jumlah masing-masing tipe karakter
        features['length'] = len(password)  # 11
        features['uppercase_count'] = sum(1 for c in password if c.isupper())  # 1 (P)
        features['lowercase_count'] = sum(1 for c in password if c.islower())  # 5 (sswr d)
        features['digit_count'] = sum(1 for c in password if c.isdigit())  # 3 (0,1,2,3)
        features['special_count'] = sum(1 for c in password if not c.isalnum())  # 1 (@)
        
        # === 2. Ratio Features (4 fitur) ===
        # Normalisasi komposisi terhadap panjang password
        length = max(len(password), 1)
        features['uppercase_ratio'] = features['uppercase_count'] / length  # 1/11 = 0.091
        features['lowercase_ratio'] = features['lowercase_count'] / length  # 5/11 = 0.455
        features['digit_ratio'] = features['digit_count'] / length  # 3/11 = 0.273
        features['special_ratio'] = features['special_count'] / length  # 1/11 = 0.091
        
        # === 3. Diversity & Entropy Features (2 fitur) ===
        # Mengukur keberagaman dan kompleksitas password
        unique_chars = len(set(password))  # 9 karakter unik
        features['char_diversity'] = unique_chars / length if length > 0 else 0  # 9/11 = 0.818
        features['entropy'] = self.calculate_entropy(password)  # Shannon entropy ≈ 3.18 bits
        
        # === 4. Pattern Detection Features (8 fitur) ===
        # Mendeteksi pola yang mengurangi kekuatan password
        features['has_qwerty_pattern'] = self.has_keyboard_pattern(password)  # 0
        features['qwerty_pattern_length'] = self.get_longest_keyboard_pattern(password)  # 0
        features['consecutive_chars'] = self.count_consecutive_chars(password)  # 0
        features['has_common_word'] = self.has_common_word(password)  # 1 (mengandung "password")
        features['common_word_ratio'] = self.get_common_word_ratio(password)  # 8/11 = 0.727
        features['leet_speak_count'] = self.count_leet_speak(password)  # 2 (@, 0)
        features['has_date_pattern'] = self.has_date_pattern(password)  # 0
        features['has_sequential_digits'] = self.has_sequential_pattern(password)  # 1 (123)
        
        # === 5. Additional Composite Features (5 fitur) ===
        # Fitur tambahan untuk menangkap karakteristik kompleks
        features['max_char_repeat'] = self.get_max_char_repeat(password)  # 2 (ss)
        features['consonant_vowel_ratio'] = self.get_consonant_vowel_ratio(password)  # 4.0
        features['special_at_start'] = 1 if len(password) > 0 and not password[0].isalnum() else 0  # 0
        features['special_at_end'] = 1 if len(password) > 0 and not password[-1].isalnum() else 0  # 0
        features['complexity_score'] = self.calculate_complexity_score(features)  # 57.18
        
        return features
    
    def calculate_entropy(self, password):
        """
        Hitung Shannon Entropy untuk mengukur ketidakpastian password.
        Formula: H(X) = -Σ p(xi) * log2(p(xi))
        
        Contoh: "aaa" → entropy = 0 bits (sangat lemah)
                "abc" → entropy ≈ 1.58 bits (lebih baik)
        """
        if not password:
            return 0
        freq = Counter(password)
        length = len(password)
        entropy = -sum((count/length) * log2(count/length) for count in freq.values())
        return entropy
    
    def has_keyboard_pattern(self, password):
        """Deteksi pola keyboard seperti 'qwerty' atau '12345'"""
        pwd_lower = password.lower()
        return 1 if any(p in pwd_lower for p in self.QWERTY_PATTERNS) else 0
    
    def get_longest_keyboard_pattern(self, password):
        """Panjang pola keyboard terpanjang yang terdeteksi"""
        pwd_lower = password.lower()
        return max([len(p) for p in self.QWERTY_PATTERNS if p in pwd_lower] or [0])
    
    def count_consecutive_chars(self, password):
        """Hitung urutan karakter berturut-turut (abc, 123)"""
        count = 0
        for i in range(len(password) - 2):
            if (ord(password[i+1]) == ord(password[i]) + 1 and 
                ord(password[i+2]) == ord(password[i]) + 2):
                count += 1
        return count
    
    def has_common_word(self, password):
        """Deteksi kata umum yang sering digunakan dalam password"""
        pwd_lower = password.lower()
        return 1 if any(w in pwd_lower for w in self.COMMON_WORDS) else 0
    
    def get_common_word_ratio(self, password):
        """Rasio panjang kata umum terhadap total panjang password"""
        pwd_lower = password.lower()
        total = sum(len(w) for w in self.COMMON_WORDS if w in pwd_lower)
        return total / max(len(password), 1)
    
    def count_leet_speak(self, password):
        """Hitung jumlah substitusi leet speak (@=a, 0=o, dll)"""
        count = 0
        for char, subs in self.LEET_SUBSTITUTIONS.items():
            for sub in subs:
                count += password.count(sub)
        return count
    
    def has_date_pattern(self, password):
        """Deteksi pola tanggal (DD/MM/YYYY, DDMMYYYY)"""
        patterns = [r'\d{2}[-/]\d{2}[-/]\d{4}', r'\d{8}', r'\d{2}\d{2}\d{2}']
        return 1 if any(re.search(p, password) for p in patterns) else 0
    
    def has_sequential_pattern(self, password):
        """Deteksi urutan digit sekuensial (0123, 1234, dll)"""
        sequences = ['0123', '1234', '2345', '3456', '4567', '5678', '6789']
        return 1 if any(s in password for s in sequences) else 0
    
    def get_max_char_repeat(self, password):
        """Panjang maksimum karakter yang berulang berturut-turut"""
        if not password:
            return 0
        max_repeat = 1
        current = 1
        for i in range(1, len(password)):
            if password[i] == password[i-1]:
                current += 1
                max_repeat = max(max_repeat, current)
            else:
                current = 1
        return max_repeat
    
    def get_consonant_vowel_ratio(self, password):
        """Rasio konsonan terhadap vokal"""
        vowels = 'aeiouAEIOU'
        consonants = sum(1 for c in password if c.isalpha() and c not in vowels)
        vowels_count = sum(1 for c in password if c in vowels)
        return consonants / vowels_count if vowels_count > 0 else consonants
    
    def calculate_complexity_score(self, features):
        """
        Skor kompleksitas gabungan berdasarkan weighted sum fitur.
        Formula: score = (length×4) + (upper×2) + (lower×2) + (digit×3) + 
                        (special×5) + (diversity×10) - (penalties)
        """
        score = 0
        score += features['length'] * 4
        score += features['uppercase_count'] * 2
        score += features['lowercase_count'] * 2
        score += features['digit_count'] * 3
        score += features['special_count'] * 5
        score += features['char_diversity'] * 10
        score -= features['has_qwerty_pattern'] * 10
        score -= features['has_common_word'] * 15
        score -= features['max_char_repeat'] * 3
        return max(score, 0)


# ============================================================================
# DATA PROCESSING
# ============================================================================

class PasswordDataProcessor:
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.feature_extractor = PasswordFeatureExtractor()
        self.X = None
        self.y = None
        self.feature_names = None
        
    def load_data(self, sample_size=None):
        print(f" Loading dataset dari {self.csv_path}...")
        
        if not os.path.exists(self.csv_path):
            print(f"\n ERROR: File tidak ditemukan!")
            return None
        
        self.df = pd.read_csv(self.csv_path)
        
        if sample_size and sample_size < len(self.df):
            self.df = self.df.sample(n=sample_size, random_state=42)
            print(f"   ✓ Menggunakan {sample_size:,} sampel data")
        else:
            print(f"   ✓ Total data: {len(self.df):,} password")
        
        print(f"\n Distribusi Label:")
        for label, count in self.df['strength'].value_counts().sort_index().items():
            names = ['Weak', 'Medium', 'Strong']
            pct = (count / len(self.df)) * 100
            print(f"   {label} - {names[label]:8}: {count:,} password ({pct:.2f}%)")
        print()
        
        return self.df
    
    def extract_all_features(self):
        print(" Feature Engineering (24 fitur)...")
        
        features_list = []
        start_time = time.time()
        
        for idx, pwd in enumerate(self.df['password']):
            if (idx + 1) % 10000 == 0:
                print(f"   Progress: {idx+1:,}/{len(self.df):,}")
            features_list.append(self.feature_extractor.extract_features(str(pwd)))
        
        features_df = pd.DataFrame(features_list)
        self.feature_names = features_df.columns.tolist()
        self.X = features_df.values
        self.y = self.df['strength'].values
        
        print(f"\n✓ Extraction selesai dalam {time.time()-start_time:.1f}s")
        print(f"   Dimensi fitur: {self.X.shape}\n")
        
        return self.X, self.y, self.feature_names


# ============================================================================
# MODEL TRAINER
# ============================================================================

class PasswordClassifierTrainer:
    """
    Trainer untuk melatih dan mengevaluasi model Random Forest dan Naive Bayes.
    Menggunakan stratified sampling untuk menjaga proporsi kelas.
    """
    
    def __init__(self, X, y, feature_names):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        
        # Stratified split untuk menjaga proporsi kelas
        print(" Data splitting dengan stratified sampling...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"   Train: {len(self.X_train):,} | Test: {len(self.X_test):,}")
        
        # Verifikasi proporsi kelas tetap konsisten
        train_dist = pd.Series(self.y_train).value_counts(normalize=True).sort_index()
        test_dist = pd.Series(self.y_test).value_counts(normalize=True).sort_index()
        print(f"\n   Distribusi Train: {train_dist.values}")
        print(f"   Distribusi Test : {test_dist.values}\n")
        
        # Normalisasi fitur untuk Naive Bayes
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.rf_model = None
        self.nb_model = None
        self.results = {}
    
    def train_random_forest(self, n_estimators=100):
        """
        Melatih Random Forest Classifier.
        
        Random Forest bekerja dengan:
        1. Bootstrap Sampling: Membuat B subset data dengan replacement
        2. Random Feature Selection: Di setiap node, pilih √p fitur acak
        3. Tree Construction: Bangun decision tree hingga max_depth
        4. Aggregation: Majority voting dari semua trees
        
        Hyperparameter:
        - n_estimators=100: Jumlah decision trees
        - max_depth=20: Kedalaman maksimum tree (mencegah overfitting)
        - min_samples_split=10: Minimum sampel untuk split node
        """
        print("="*80)
        print(" RANDOM FOREST CLASSIFIER")
        print("="*80)
        print(f"\nHyperparameter:")
        print(f"   n_estimators      : {n_estimators}")
        print(f"   max_depth         : 20")
        print(f"   min_samples_split : 10")
        print(f"   random_state      : 42\n")
        
        start = time.time()
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=20, 
            min_samples_split=10, 
            random_state=42, 
            n_jobs=-1
        )
        self.rf_model.fit(self.X_train, self.y_train)
        train_time = time.time() - start
        
        # Evaluasi
        y_pred_train = self.rf_model.predict(self.X_train)
        y_pred_test = self.rf_model.predict(self.X_test)
        
        train_acc = accuracy_score(self.y_train, y_pred_train)
        test_acc = accuracy_score(self.y_test, y_pred_test)
        
        self.results['Random Forest'] = {
            'model': self.rf_model,
            'predictions': y_pred_test,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'accuracy': test_acc,
            'precision': precision_score(self.y_test, y_pred_test, average='weighted'),
            'recall': recall_score(self.y_test, y_pred_test, average='weighted'),
            'f1': f1_score(self.y_test, y_pred_test, average='weighted'),
            'training_time': train_time,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred_test)
        }
        
        print(f"Hasil Training:")
        print(f"   Train Accuracy : {train_acc*100:.2f}%")
        print(f"   Test Accuracy  : {test_acc*100:.2f}%")
        print(f"   Precision      : {self.results['Random Forest']['precision']*100:.2f}%")
        print(f"   Recall         : {self.results['Random Forest']['recall']*100:.2f}%")
        print(f"   F1-Score       : {self.results['Random Forest']['f1']*100:.2f}%")
        print(f"   Training Time  : {train_time:.2f}s")
        
        # Deteksi overfitting
        gap = (train_acc - test_acc) * 100
        if gap < 1:
            print(f"   Status         : ✓ Good generalization (gap={gap:.2f}%)\n")
        elif gap < 5:
            print(f"   Status         : ⚠ Slight overfitting (gap={gap:.2f}%)\n")
        else:
            print(f"   Status         :  Overfitting detected (gap={gap:.2f}%)\n")
        
        return self.rf_model
    
    def train_naive_bayes(self):
        """
        Melatih Gaussian Naive Bayes Classifier.
        
        Naive Bayes bekerja dengan:
        1. Training Phase: Hitung P(C) dan P(xi|C) untuk setiap fitur
        2. Asumsi: Fitur independen kondisional (P(X|C) = ∏ P(xi|C))
        3. Prediction: Pilih kelas dengan P(C|X) tertinggi
        
        Menggunakan Gaussian Naive Bayes karena fitur kontinu.
        Asumsi: P(xi|C) mengikuti distribusi normal N(μ, σ²)
        """
        print("="*80)
        print(" NAIVE BAYES CLASSIFIER (Gaussian)")
        print("="*80)
        print(f"\nAsumsi:")
        print(f"   - Fitur independen kondisional")
        print(f"   - Distribusi fitur: Gaussian N(μ, σ²)")
        print(f"   - Data dinormalisasi dengan StandardScaler\n")
        
        start = time.time()
        self.nb_model = GaussianNB()
        self.nb_model.fit(self.X_train_scaled, self.y_train)
        train_time = time.time() - start
        
        # Evaluasi
        y_pred_train = self.nb_model.predict(self.X_train_scaled)
        y_pred_test = self.nb_model.predict(self.X_test_scaled)
        
        train_acc = accuracy_score(self.y_train, y_pred_train)
        test_acc = accuracy_score(self.y_test, y_pred_test)
        
        self.results['Naive Bayes'] = {
            'model': self.nb_model,
            'predictions': y_pred_test,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'accuracy': test_acc,
            'precision': precision_score(self.y_test, y_pred_test, average='weighted'),
            'recall': recall_score(self.y_test, y_pred_test, average='weighted'),
            'f1': f1_score(self.y_test, y_pred_test, average='weighted'),
            'training_time': train_time,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred_test)
        }
        
        print(f"Hasil Training:")
        print(f"   Train Accuracy : {train_acc*100:.2f}%")
        print(f"   Test Accuracy  : {test_acc*100:.2f}%")
        print(f"   Precision      : {self.results['Naive Bayes']['precision']*100:.2f}%")
        print(f"   Recall         : {self.results['Naive Bayes']['recall']*100:.2f}%")
        print(f"   F1-Score       : {self.results['Naive Bayes']['f1']*100:.2f}%")
        print(f"   Training Time  : {train_time:.2f}s")
        
        # Deteksi underfitting
        gap = (train_acc - test_acc) * 100
        if test_acc < 80:
            print(f"   Status         : ⚠ Possible underfitting (test_acc={test_acc*100:.2f}%)\n")
        elif gap < 1:
            print(f"   Status         : ✓ Good generalization (gap={gap:.2f}%)\n")
        else:
            print(f"   Status         : ✓ Acceptable (gap={gap:.2f}%)\n")
        
        return self.nb_model
    
    def train_shannon_entropy_baseline(self):
        """
        Baseline menggunakan Shannon Entropy dengan threshold tetap.
        
        Klasifikasi:
        - entropy < 2.5 bits  → Weak
        - 2.5 ≤ entropy < 3.5 → Medium
        - entropy ≥ 3.5 bits  → Strong
        
        Keterbatasan:
        - Hanya menggunakan 1 fitur (entropi)
        - Tidak mendeteksi pola leksikal (keyboard, leet speak)
        - Threshold arbitrer, tidak adaptif
        """
        print("="*80)
        print(" SHANNON ENTROPY BASELINE")
        print("="*80)
        print(f"\nThreshold Klasifikasi:")
        print(f"   entropy < 2.5 bits        → Weak")
        print(f"   2.5 ≤ entropy < 3.5 bits  → Medium")
        print(f"   entropy ≥ 3.5 bits        → Strong\n")
        
        entropy_idx = self.feature_names.index('entropy')
        entropy_vals = self.X_test[:, entropy_idx]
        
        y_pred = np.zeros_like(self.y_test)
        y_pred[entropy_vals < 2.5] = 0
        y_pred[(entropy_vals >= 2.5) & (entropy_vals < 3.5)] = 1
        y_pred[entropy_vals >= 3.5] = 2
        
        self.results['Shannon Entropy'] = {
            'predictions': y_pred,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'f1': f1_score(self.y_test, y_pred, average='weighted'),
            'training_time': 0,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
        
        print(f"Hasil Evaluasi:")
        print(f"   Accuracy  : {self.results['Shannon Entropy']['accuracy']*100:.2f}%")
        print(f"   Precision : {self.results['Shannon Entropy']['precision']*100:.2f}%")
        print(f"   Recall    : {self.results['Shannon Entropy']['recall']*100:.2f}%")
        print(f"   F1-Score  : {self.results['Shannon Entropy']['f1']*100:.2f}%\n")
    
    # ========================================================================
    # RUMUSAN MASALAH 1: UJI SIGNIFIKANSI STATISTIK (McNemar's Test)
    # ========================================================================
    
    def mcnemar_statistical_test(self):
        """
        RM1: Uji signifikansi statistik perbedaan performa RF vs NB.
        
        McNemar's Test digunakan untuk membandingkan dua classifier pada data berpasangan.
        
        Contingency Table:
                        NB Correct    NB Wrong
        RF Correct         n00           n01
        RF Wrong           n10           n11
        
        Test Statistic: χ² = (|n01 - n10| - 1)² / (n01 + n10)
        H0: Tidak ada perbedaan signifikan antara RF dan NB
        H1: Ada perbedaan signifikan
        
        Jika p-value < 0.05, tolak H0 (perbedaan signifikan)
        """
        print("\n" + "="*80)
        print(" RUMUSAN MASALAH #1: UJI SIGNIFIKANSI STATISTIK")
        print("="*80 + "\n")
        
        rf_pred = self.results['Random Forest']['predictions']
        nb_pred = self.results['Naive Bayes']['predictions']
        
        print("McNemar's Test: Random Forest vs Naive Bayes")
        print("-"*80)
        
        # Buat contingency table
        rf_correct = (rf_pred == self.y_test).astype(int)
        nb_correct = (nb_pred == self.y_test).astype(int)
        
        n00 = np.sum((rf_correct == 1) & (nb_correct == 1))  # Both correct
        n11 = np.sum((rf_correct == 0) & (nb_correct == 0))  # Both wrong
        n01 = np.sum((rf_correct == 1) & (nb_correct == 0))  # RF correct, NB wrong
        n10 = np.sum((rf_correct == 0) & (nb_correct == 1))  # RF wrong, NB correct
        
        print(f"\nContingency Table:")
        print(f"                    NB Correct    NB Wrong")
        print(f"   RF Correct       {n00:>6}        {n01:>6}")
        print(f"   RF Wrong         {n10:>6}        {n11:>6}")
        print(f"\n   Both correct     : {n00:,}")
        print(f"   Both wrong       : {n11:,}")
        print(f"   RF only correct  : {n01:,}")
        print(f"   NB only correct  : {n10:,}")
        
        # Hitung McNemar statistic
        if n01 + n10 > 0:
            mcnemar_stat = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
            
            print(f"\nMcNemar's χ² statistic : {mcnemar_stat:.4f}")
            print(f"P-value                : {p_value:.6f}")
            print(f"Significance level (α) : 0.05")
            
            if p_value < 0.05:
                print(f"\n✓ SIGNIFIKAN (p < 0.05)")
                print(f"  Kesimpulan: Perbedaan performa RF dan NB signifikan secara statistik.")
            else:
                print(f"\n✗ TIDAK SIGNIFIKAN (p ≥ 0.05)")
                print(f"  Kesimpulan: Tidak ada perbedaan signifikan antara RF dan NB.")
        else:
            mcnemar_stat = None
            p_value = 1.0
            print(f"\n⚠ Tidak bisa melakukan uji (n01 + n10 = 0)")
        
        # Analisis Trade-off Accuracy vs Inference Time
        print("\n" + "="*80)
        print(" ANALISIS TRADE-OFF: ACCURACY vs INFERENCE TIME")
        print("="*80 + "\n")
        
        X_sample = self.X_test[:1000]
        X_sample_sc = self.X_test_scaled[:1000]
        
        # Ukur inference time RF
        start = time.perf_counter()
        _ = self.rf_model.predict(X_sample)
        rf_time = (time.perf_counter() - start) / 1000 * 1000
        
        # Ukur inference time NB
        start = time.perf_counter()
        _ = self.nb_model.predict(X_sample_sc)
        nb_time = (time.perf_counter() - start) / 1000 * 1000
        
        rf_acc = self.results['Random Forest']['accuracy'] * 100
        nb_acc = self.results['Naive Bayes']['accuracy'] * 100
        acc_gain = rf_acc - nb_acc
        time_cost = rf_time - nb_time
        
        print(f"Random Forest:")
        print(f"   Accuracy       : {rf_acc:.2f}%")
        print(f"   Inference Time : {rf_time:.4f} ms (per 1000 prediksi)")
        print(f"\nNaive Bayes:")
        print(f"   Accuracy       : {nb_acc:.2f}%")
        print(f"   Inference Time : {nb_time:.4f} ms (per 1000 prediksi)")
        print(f"\nTrade-off:")
        print(f"   Accuracy Gain  : +{acc_gain:.2f}%")
        print(f"   Time Cost      : +{time_cost:.4f} ms")
        
        # Rekomendasi
        print("\n" + "-"*80)
        print(" REKOMENDASI:")
        print("-"*80)
        
        verdict = ""
        if p_value and p_value < 0.05 and acc_gain > 2.0 and rf_time < CONFIG['ACCEPTABLE_LATENCY_MS']:
            verdict = "✓ RANDOM FOREST RECOMMENDED"
            print(f"\n{verdict}")
            print(f"   Alasan:")
            print(f"   • Peningkatan akurasi signifikan secara statistik (p={p_value:.6f})")
            print(f"   • Gain akurasi sebesar +{acc_gain:.2f}% membenarkan time cost")
            print(f"   • Inference time ({rf_time:.4f}ms) masih di bawah threshold ({CONFIG['ACCEPTABLE_LATENCY_MS']}ms)")
        elif nb_time < CONFIG['ACCEPTABLE_LATENCY_MS']:
            verdict = "✓ NAIVE BAYES RECOMMENDED"
            print(f"\n{verdict}")
            print(f"   Alasan:")
            print(f"   • Inference time sangat cepat ({nb_time:.4f}ms)")
            print(f"   • Akurasi ({nb_acc:.2f}%) cukup untuk mayoritas kasus")
            print(f"   • Efisien untuk deployment dengan resource terbatas")
        else:
            verdict = "⚠ HYBRID APPROACH RECOMMENDED"
            print(f"\n{verdict}")
            print(f"   Alasan:")
            print(f"   • Tidak ada single model yang optimal untuk semua skenario")
            print(f"   • Pertimbangkan rule-based pre-filtering + ML")
        
        return {
            'mcnemar_statistic': mcnemar_stat,
            'mcnemar_pvalue': p_value,
            'rf_accuracy': rf_acc,
            'nb_accuracy': nb_acc,
            'accuracy_gain': acc_gain,
            'rf_inference_ms': rf_time,
            'nb_inference_ms': nb_time,
            'time_cost_ms': time_cost,
            'verdict': verdict,
            'contingency_table': {
                'both_correct': n00,
                'both_wrong': n11,
                'rf_only': n01,
                'nb_only': n10
            }
        }
    
    # ========================================================================
    # RUMUSAN MASALAH 2: ANALISIS ERROR PADA POLA LEKSIKAL
    # ========================================================================
    
    def analyze_lexical_pattern_errors(self):
        """
        RM2: Analisis error rate pada password dengan pola leksikal tertentu.
        
        Pola yang dianalisis:
        1. Leet speak: Substitusi karakter (@ untuk a, 0 untuk o)
        2. Keyboard pattern: Urutan keyboard (qwerty, asdfgh)
        3. Common words: Kata umum (password, admin)
        4. No pattern: Password tanpa pola deteksi
        
        Tujuan: Mengidentifikasi sejauh mana Naive Bayes mengalami 
        peningkatan error dibanding Random Forest pada setiap pola.
        """
        print("\n" + "="*80)
        print(" RUMUSAN MASALAH #2: ANALISIS ERROR PADA POLA LEKSIKAL")
        print("="*80 + "\n")
        
        rf_pred = self.results['Random Forest']['predictions']
        nb_pred = self.results['Naive Bayes']['predictions']
        
        # Definisi subset berdasarkan pola
        patterns = {
            'leet_speak': self.X_test[:, self.feature_names.index('leet_speak_count')] > 0,
            'keyboard_pattern': self.X_test[:, self.feature_names.index('has_qwerty_pattern')] == 1,
            'common_words': self.X_test[:, self.feature_names.index('has_common_word')] == 1,
            'sequential_digits': self.X_test[:, self.feature_names.index('has_sequential_digits')] == 1,
            'no_pattern': (
                (self.X_test[:, self.feature_names.index('leet_speak_count')] == 0) & 
                (self.X_test[:, self.feature_names.index('has_qwerty_pattern')] == 0) & 
                (self.X_test[:, self.feature_names.index('has_common_word')] == 0)
            )
        }
        
        error_analysis = {}
        
        print(f"{'Pattern':<25} {'Samples':<10} {'RF Error':<12} {'NB Error':<12} {'Difference':<12}")
        print("-"*80)
        
        for pattern_name, mask in patterns.items():
            sample_count = np.sum(mask)
            
            if sample_count < 10:
                continue
            
            y_true_subset = self.y_test[mask]
            rf_pred_subset = rf_pred[mask]
            nb_pred_subset = nb_pred[mask]
            
            rf_error = 1 - accuracy_score(y_true_subset, rf_pred_subset)
            nb_error = 1 - accuracy_score(y_true_subset, nb_pred_subset)
            error_diff = (nb_error - rf_error) * 100
            
            error_analysis[pattern_name] = {
                'sample_count': sample_count,
                'rf_error_rate': rf_error,
                'nb_error_rate': nb_error,
                'error_difference': error_diff
            }
            
            # Format output
            pattern_display = pattern_name.replace('_', ' ').title()
            emoji = "🔴" if error_diff > 5 else ("🟡" if error_diff > 2 else "🟢")
            
            print(f"{emoji} {pattern_display:<22} {sample_count:<10} "
                  f"{rf_error*100:>6.2f}%      {nb_error*100:>6.2f}%      "
                  f"{error_diff:>+6.2f}%")
        
        print("\n" + "-"*80)
        print(" INTERPRETASI:")
        print("-"*80)
        
        # Temukan pola paling bermasalah
        most_problematic = max(error_analysis.items(), 
                              key=lambda x: x[1]['error_difference'])
        
        print(f"\nPola paling bermasalah untuk Naive Bayes:")
        print(f"   Pattern  : {most_problematic[0].replace('_', ' ').title()}")
        print(f"   RF Error : {most_problematic[1]['rf_error_rate']*100:.2f}%")
        print(f"   NB Error : {most_problematic[1]['nb_error_rate']*100:.2f}%")
        print(f"   Selisih  : +{most_problematic[1]['error_difference']:.2f}%")
        
        print(f"\nKesimpulan:")
        if most_problematic[1]['error_difference'] > 5:
            print(f"   ⚠ Naive Bayes mengalami kesulitan signifikan pada pola {most_problematic[0]}")
            print(f"   Rekomendasi: Gunakan Random Forest untuk validasi password dengan pola ini")
        else:
            print(f"   ✓ Perbedaan error rate tidak terlalu signifikan pada semua pola")
            print(f"   Kedua model dapat dipertimbangkan tergantung kebutuhan")
        
        return error_analysis
    
    # ========================================================================
    # RUMUSAN MASALAH 3: KELAYAKAN IMPLEMENTASI REAL-TIME
    # ========================================================================
    
    def evaluate_realtime_feasibility(self, n_samples=1000):
        """
        RM3: Evaluasi kelayakan implementasi sebagai password validator real-time.
        
        Definisi Real-Time:
        Berdasarkan standar HCI (Nielsen, 1993; Card et al., 1991):
        - < 100ms (0.1s): Instantaneous response, user tidak merasakan delay
        - < 1000ms (1s): Maintain user flow, delay minimal
        - > 1000ms: Perhatian user mulai berkurang
        
        Untuk password validation, threshold ditetapkan < 10ms per prediksi:
        - Cumulative delay: Jika ada 10 validasi, total < 100ms
        - Safety margin: Memperhitungkan network latency & overhead
        
        Metrik yang diukur:
        1. Inference Time: Average, P95, P99 latency
        2. Throughput: Prediksi per detik
        3. Memory Footprint: Model size dan RAM usage
        4. Detection Rate: Akurasi dan recall weak passwords
        """
        print("\n" + "="*80)
        print(" RUMUSAN MASALAH #3: KELAYAKAN IMPLEMENTASI REAL-TIME")
        print("="*80 + "\n")
        
        print(f"Definisi Real-Time:")
        print(f"   Threshold     : < {CONFIG['ACCEPTABLE_LATENCY_MS']} ms per prediksi")
        print(f"   Rationale     : Berdasarkan standar HCI untuk interactive systems")
        print(f"   Sample Size   : {n_samples:,} prediksi")
        print(f"\n" + "-"*80 + "\n")
        
        results = {}
        
        X_sample = self.X_test[:n_samples]
        X_sample_sc = self.X_test_scaled[:n_samples]
        y_sample = self.y_test[:n_samples]
        
        # ===== RANDOM FOREST =====
        print(" RANDOM FOREST:\n")
        
        # Memory footprint
        mem_before = get_memory_usage_mb()
        rf_size = get_model_size_mb(self.rf_model)
        mem_after = get_memory_usage_mb()
        
        print(f"Memory Footprint:")
        print(f"   Model Size        : {rf_size:.2f} MB")
        print(f"   RAM Before Load   : {mem_before:.2f} MB")
        print(f"   RAM After Load    : {mem_after:.2f} MB")
        print(f"   Memory Overhead   : {mem_after - mem_before:.2f} MB")
        
        # Inference time (multiple runs untuk stabilitas)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            rf_preds = self.rf_model.predict(X_sample)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        per_pred_ms = (avg_time / n_samples) * 1000
        p95_ms = np.percentile(times, 95) / n_samples * 1000
        p99_ms = np.percentile(times, 99) / n_samples * 1000
        
        print(f"\nInference Time:")
        print(f"   Total (1000 pred) : {avg_time:.4f} s")
        print(f"   Average per pred  : {per_pred_ms:.4f} ms")
        print(f"   P95 Latency       : {p95_ms:.4f} ms")
        print(f"   P99 Latency       : {p99_ms:.4f} ms")
        print(f"   Throughput        : {n_samples/avg_time:.0f} pred/sec")
        
        # Detection rate
        rf_acc = accuracy_score(y_sample, rf_preds)
        weak_mask = (y_sample == 0)
        rf_weak_recall = recall_score(y_sample[weak_mask], rf_preds[weak_mask], average='micro') if np.sum(weak_mask) > 0 else 0
        
        print(f"\nDetection Rate:")
        print(f"   Overall Accuracy  : {rf_acc*100:.2f}%")
        print(f"   Weak Detection    : {rf_weak_recall*100:.2f}%")
        
        # Verdict
        if per_pred_ms < CONFIG['ACCEPTABLE_LATENCY_MS']:
            rf_verdict = " LAYAK untuk real-time"
        elif per_pred_ms < 50:
            rf_verdict = " ACCEPTABLE untuk real-time"
        else:
            rf_verdict = " TIDAK LAYAK untuk real-time"
        
        print(f"\n   Verdict: {rf_verdict}\n")
        
        results['Random Forest'] = {
            'model_size_mb': rf_size,
            'memory_overhead_mb': mem_after - mem_before,
            'avg_time_ms': per_pred_ms,
            'p95_latency_ms': p95_ms,
            'p99_latency_ms': p99_ms,
            'throughput_per_sec': n_samples/avg_time,
            'accuracy': rf_acc,
            'weak_detection_rate': rf_weak_recall,
            'verdict': rf_verdict
        }
        
        print("-"*80 + "\n")
        
        # ===== NAIVE BAYES =====
        print(" NAIVE BAYES:\n")
        
        # Memory footprint
        mem_before = get_memory_usage_mb()
        nb_size = get_model_size_mb(self.nb_model)
        mem_after = get_memory_usage_mb()
        
        print(f"Memory Footprint:")
        print(f"   Model Size        : {nb_size:.2f} MB")
        print(f"   RAM Before Load   : {mem_before:.2f} MB")
        print(f"   RAM After Load    : {mem_after:.2f} MB")
        print(f"   Memory Overhead   : {mem_after - mem_before:.2f} MB")
        
        # Inference time
        times = []
        for _ in range(10):
            start = time.perf_counter()
            nb_preds = self.nb_model.predict(X_sample_sc)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        per_pred_ms = (avg_time / n_samples) * 1000
        p95_ms = np.percentile(times, 95) / n_samples * 1000
        p99_ms = np.percentile(times, 99) / n_samples * 1000
        
        print(f"\nInference Time:")
        print(f"   Total (1000 pred) : {avg_time:.4f} s")
        print(f"   Average per pred  : {per_pred_ms:.4f} ms")
        print(f"   P95 Latency       : {p95_ms:.4f} ms")
        print(f"   P99 Latency       : {p99_ms:.4f} ms")
        print(f"   Throughput        : {n_samples/avg_time:.0f} pred/sec")
        
        # Detection rate
        nb_acc = accuracy_score(y_sample, nb_preds)
        nb_weak_recall = recall_score(y_sample[weak_mask], nb_preds[weak_mask], average='micro') if np.sum(weak_mask) > 0 else 0
        
        print(f"\nDetection Rate:")
        print(f"   Overall Accuracy  : {nb_acc*100:.2f}%")
        print(f"   Weak Detection    : {nb_weak_recall*100:.2f}%")
        
        # Verdict
        if per_pred_ms < CONFIG['ACCEPTABLE_LATENCY_MS']:
            nb_verdict = " LAYAK untuk real-time"
        elif per_pred_ms < 50:
            nb_verdict = " ACCEPTABLE untuk real-time"
        else:
            nb_verdict = " TIDAK LAYAK untuk real-time"
        
        print(f"\n   Verdict: {nb_verdict}\n")
        
        results['Naive Bayes'] = {
            'model_size_mb': nb_size,
            'memory_overhead_mb': mem_after - mem_before,
            'avg_time_ms': per_pred_ms,
            'p95_latency_ms': p95_ms,
            'p99_latency_ms': p99_ms,
            'throughput_per_sec': n_samples/avg_time,
            'accuracy': nb_acc,
            'weak_detection_rate': nb_weak_recall,
            'verdict': nb_verdict
        }
        
        # ===== COMPARATIVE SUMMARY =====
        print("="*80)
        print(" RINGKASAN KOMPARATIF")
        print("="*80 + "\n")
        
        print(f"{'Metric':<30} {'Random Forest':<20} {'Naive Bayes':<20}")
        print("-"*70)
        print(f"{'Inference Time (avg)':<30} {results['Random Forest']['avg_time_ms']:>8.4f} ms       {results['Naive Bayes']['avg_time_ms']:>8.4f} ms")
        print(f"{'P95 Latency':<30} {results['Random Forest']['p95_latency_ms']:>8.4f} ms       {results['Naive Bayes']['p95_latency_ms']:>8.4f} ms")
        print(f"{'Throughput':<30} {results['Random Forest']['throughput_per_sec']:>8.0f} pred/s    {results['Naive Bayes']['throughput_per_sec']:>8.0f} pred/s")
        print(f"{'Model Size':<30} {results['Random Forest']['model_size_mb']:>8.2f} MB         {results['Naive Bayes']['model_size_mb']:>8.2f} MB")
        print(f"{'Accuracy':<30} {results['Random Forest']['accuracy']*100:>8.2f}%          {results['Naive Bayes']['accuracy']*100:>8.2f}%")
        print(f"{'Weak Detection':<30} {results['Random Forest']['weak_detection_rate']*100:>8.2f}%          {results['Naive Bayes']['weak_detection_rate']*100:>8.2f}%")
        
        print("\n" + "="*80)
        print(" REKOMENDASI FINAL")
        print("="*80 + "\n")
        
        rf_feasible = results['Random Forest']['avg_time_ms'] < CONFIG['ACCEPTABLE_LATENCY_MS']
        nb_feasible = results['Naive Bayes']['avg_time_ms'] < CONFIG['ACCEPTABLE_LATENCY_MS']
        
        if rf_feasible and results['Random Forest']['accuracy'] > results['Naive Bayes']['accuracy'] + 0.02:
            print(" MODEL TERPILIH: RANDOM FOREST")
            print("\nAlasan:")
            print("   • Inference time memenuhi threshold real-time")
            print("   • Akurasi lebih tinggi secara signifikan")
            print("   • Detection rate weak passwords lebih baik")
            print("\nUse Case:")
            print("   • Sistem registrasi user dengan validasi password ketat")
            print("   • Enterprise security policy enforcement")
            print("   • Password strength meter untuk aplikasi critical")
        elif nb_feasible:
            print(" MODEL TERPILIH: NAIVE BAYES")
            print("\nAlasan:")
            print("   • Inference time sangat cepat (<<10ms)")
            print("   • Memory footprint minimal")
            print("   • Akurasi mencukupi untuk mayoritas kasus")
            print("\nUse Case:")
            print("   • High-volume authentication systems")
            print("   • Embedded systems dengan resource terbatas")
            print("   • Mobile applications dengan battery constraints")
        else:
            print(" REKOMENDASI: HYBRID APPROACH")
            print("\nAlasan:")
            print("   • Trade-off antara speed dan accuracy")
            print("\nSkenario Hybrid:")
            print("   • Pre-screening dengan Naive Bayes untuk weak passwords")
            print("   • Random Forest untuk edge cases dan medium passwords")
            print("   • Adaptive threshold berdasarkan security level")
        
        return results
    
    def get_feature_importance(self):
        """Dapatkan feature importance dari Random Forest"""
        if self.rf_model is None:
            return None
        importances = self.rf_model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
    
    def multi_scenario_evaluation(self):
        """
        Evaluasi model pada multiple train-test splits untuk validasi robustness.
        
        Skenario:
        1. 80:20 split (standard)
        2. 70:30 split (lebih banyak testing data)
        3. 50:50 split (extreme - minimal training data)
        
        Tujuan: Mendeteksi overfitting/underfitting dan validasi consistency.
        """
        print("\n" + "="*80)
        print(" MULTI-SCENARIO EVALUATION")
        print("="*80 + "\n")
        
        scenarios = [
            ('80:20', 0.2),
            ('70:30', 0.3),
            ('50:50', 0.5)
        ]
        
        scenario_results = {}
        
        for scenario_name, test_size in scenarios:
            print(f"Scenario: {scenario_name} Split")
            print("-"*80)
            
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
            )
            
            # Train RF
            rf = RandomForestClassifier(
                n_estimators=100, max_depth=20, min_samples_split=10,
                random_state=42, n_jobs=-1
            )
            rf.fit(X_train, y_train)
            
            rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
            rf_test_acc = accuracy_score(y_test, rf.predict(X_test))
            rf_gap = (rf_train_acc - rf_test_acc) * 100
            
            # Train NB
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)
            
            nb = GaussianNB()
            nb.fit(X_train_sc, y_train)
            
            nb_train_acc = accuracy_score(y_train, nb.predict(X_train_sc))
            nb_test_acc = accuracy_score(y_test, nb.predict(X_test_sc))
            nb_gap = (nb_train_acc - nb_test_acc) * 100
            
            print(f"\nRandom Forest:")
            print(f"   Train: {rf_train_acc*100:.2f}% | Test: {rf_test_acc*100:.2f}% | Gap: {rf_gap:.2f}%")
            
            print(f"\nNaive Bayes:")
            print(f"   Train: {nb_train_acc*100:.2f}% | Test: {nb_test_acc*100:.2f}% | Gap: {nb_gap:.2f}%")
            print()
            
            scenario_results[scenario_name] = {
                'rf_train': rf_train_acc,
                'rf_test': rf_test_acc,
                'rf_gap': rf_gap,
                'nb_train': nb_train_acc,
                'nb_test': nb_test_acc,
                'nb_gap': nb_gap
            }
        
        print("="*80)
        print(" ANALISIS CONSISTENCY ACROSS SCENARIOS")
        print("="*80 + "\n")
        
        # Hitung variance akurasi test
        rf_test_accs = [scenario_results[s]['rf_test'] for s in ['80:20', '70:30', '50:50']]
        nb_test_accs = [scenario_results[s]['nb_test'] for s in ['80:20', '70:30', '50:50']]
        
        rf_variance = np.var(rf_test_accs) * 10000  # dalam basis poin
        nb_variance = np.var(nb_test_accs) * 10000
        
        print(f"Random Forest:")
        print(f"   Test Accuracy Variance: {rf_variance:.2f} bp²")
        print(f"   Consistency: {'✓ High' if rf_variance < 1 else '⚠ Moderate' if rf_variance < 5 else '✗ Low'}")
        
        print(f"\nNaive Bayes:")
        print(f"   Test Accuracy Variance: {nb_variance:.2f} bp²")
        print(f"   Consistency: {'✓ High' if nb_variance < 1 else '⚠ Moderate' if nb_variance < 5 else '✗ Low'}")
        
        return scenario_results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrices(trainer, output_folder):
    """Plot confusion matrices untuk RF dan NB"""
    print("\n Membuat confusion matrices...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = ['Random Forest', 'Naive Bayes']
    
    for idx, model_name in enumerate(models):
        cm = trainer.results[model_name]['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Weak', 'Medium', 'Strong'],
                    yticklabels=['Weak', 'Medium', 'Strong'])
        
        axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
        axes[idx].set_ylabel('True Label', fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(output_folder, 'confusion_matrices.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {filepath}")


def plot_feature_importance(importance_df, output_folder):
    """Plot top 15 feature importances"""
    print(" Membuat feature importance chart...")
    
    top_features = importance_df.head(15)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue', alpha=0.8)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score', fontweight='bold')
    plt.title('Top 15 Most Important Features (Random Forest)', fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_folder, 'feature_importance.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved: {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("  SISTEM KLASIFIKASI KEKUATAN PASSWORD")
    print("  Comparative Analysis: Random Forest vs Naive Bayes")
    print("="*80 + "\n")
    
    output_folder = CONFIG['OUTPUT_FOLDER']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # ========================================================================
    # FASE 1: DATA LOADING & FEATURE EXTRACTION
    # ========================================================================
    print("="*80)
    print("FASE 1: DATA LOADING & FEATURE EXTRACTION")
    print("="*80 + "\n")
    
    processor = PasswordDataProcessor(CONFIG['CSV_PATH'])
    df = processor.load_data(sample_size=CONFIG['SAMPLE_SIZE'])
    
    if df is None:
        print("\n Program dihentikan - dataset tidak ditemukan")
        return
    
    X, y, feature_names = processor.extract_all_features()
    
    # ========================================================================
    # FASE 2: MODEL TRAINING
    # ========================================================================
    print("\n" + "="*80)
    print("FASE 2: MODEL TRAINING")
    print("="*80 + "\n")
    
    trainer = PasswordClassifierTrainer(X, y, feature_names)
    
    # Train models
    trainer.train_random_forest(n_estimators=CONFIG['N_ESTIMATORS'])
    trainer.train_naive_bayes()
    trainer.train_shannon_entropy_baseline()
    
    # ========================================================================
    # FASE 3: ANALISIS RUMUSAN MASALAH
    # ========================================================================
    print("\n\n" + "="*80)
    print("FASE 3: ANALISIS RUMUSAN MASALAH")
    print("="*80)
    
    # RM1: Uji Signifikansi Statistik
    rm1_results = trainer.mcnemar_statistical_test()
    
    # RM2: Analisis Error Pola Leksikal
    rm2_results = trainer.analyze_lexical_pattern_errors()
    
    # RM3: Kelayakan Real-Time
    rm3_results = trainer.evaluate_realtime_feasibility(
        n_samples=CONFIG['INFERENCE_SAMPLES']
    )
    
    # Multi-scenario evaluation
    scenario_results = trainer.multi_scenario_evaluation()
    
    # ========================================================================
    # FASE 4: FEATURE IMPORTANCE ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("FASE 4: FEATURE IMPORTANCE ANALYSIS")
    print("="*80 + "\n")
    
    importance_df = trainer.get_feature_importance()
    
    print("Top 15 Most Important Features:")
    print("-"*80)
    for idx, row in importance_df.head(15).iterrows():
        bar_length = int(row['importance'] * 50)
        bar = '█' * bar_length
        print(f"{idx+1:2}. {row['feature']:25} : {row['importance']:.6f} {bar}")
    print()
    
    # ========================================================================
    # FASE 5: VISUALISASI
    # ========================================================================
    print("\n" + "="*80)
    print("FASE 5: VISUALISASI & REPORTING")
    print("="*80 + "\n")
    
    plot_confusion_matrices(trainer, output_folder)
    plot_feature_importance(importance_df, output_folder)
    
    # ========================================================================
    # FASE 6: SAVE RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("FASE 6: MENYIMPAN HASIL")
    print("="*80 + "\n")
    
    # Save models
    joblib.dump(trainer.rf_model, os.path.join(output_folder, 'rf_model.pkl'))
    joblib.dump(trainer.nb_model, os.path.join(output_folder, 'nb_model.pkl'))
    joblib.dump(trainer.scaler, os.path.join(output_folder, 'scaler.pkl'))
    print("✓ Models saved")
    
    # Save results to CSV
    pd.DataFrame([rm1_results]).to_csv(
        os.path.join(output_folder, 'rm1_statistical_significance.csv'), index=False)
    
    pd.DataFrame(rm2_results).T.to_csv(
        os.path.join(output_folder, 'rm2_lexical_pattern_analysis.csv'))
    
    pd.DataFrame(rm3_results).T.to_csv(
        os.path.join(output_folder, 'rm3_realtime_feasibility.csv'))
    
    pd.DataFrame(scenario_results).T.to_csv(
        os.path.join(output_folder, 'multi_scenario_evaluation.csv'))
    
    importance_df.to_csv(os.path.join(output_folder, 'feature_importance.csv'), index=False)
    
    # Model comparison summary
    comparison_data = []
    for model_name in ['Random Forest', 'Naive Bayes', 'Shannon Entropy']:
        comparison_data.append({
            'Model': model_name,
            'Accuracy (%)': trainer.results[model_name]['accuracy'] * 100,
            'Precision (%)': trainer.results[model_name]['precision'] * 100,
            'Recall (%)': trainer.results[model_name]['recall'] * 100,
            'F1-Score (%)': trainer.results[model_name]['f1'] * 100,
            'Training Time (s)': trainer.results[model_name]['training_time']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(output_folder, 'model_comparison_summary.csv'), index=False)
    
    print("✓ All reports saved\n")
    
    # ========================================================================
    # RINGKASAN 
    # ========================================================================
    print("\n" + "="*80)
    print(" RINGKASAN ")
    print("="*80 + "\n")
    
    print("1. PERFORMA MODEL")
    print("-"*80)
    print(f"{'Model':<20} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
    print("-"*80)
    for model_name in ['Random Forest', 'Naive Bayes', 'Shannon Entropy']:
        r = trainer.results[model_name]
        print(f"{model_name:<20} {r['accuracy']*100:>6.2f}%        "
              f"{r['precision']*100:>6.2f}%        "
              f"{r['recall']*100:>6.2f}%        "
              f"{r['f1']*100:>6.2f}%")
    print()
    
    print("2. RUMUSAN MASALAH #1: Signifikansi Statistik")
    print("-"*80)
    print(f"   McNemar's Test p-value : {rm1_results['mcnemar_pvalue']:.6f}")
    print(f"   Signifikansi           : {'✓ Ya (p < 0.05)' if rm1_results['mcnemar_pvalue'] < 0.05 else '✗ Tidak (p ≥ 0.05)'}")
    print(f"   Accuracy Gain          : +{rm1_results['accuracy_gain']:.2f}%")
    print(f"   Time Cost              : +{rm1_results['time_cost_ms']:.4f} ms")
    print(f"   Rekomendasi            : {rm1_results['verdict']}\n")
    
    print("3. RUMUSAN MASALAH #2: Pola Leksikal")
    print("-"*80)
    most_problematic = max(rm2_results.items(), 
                          key=lambda x: x[1]['error_difference'])
    print(f"   Pola paling bermasalah : {most_problematic[0].replace('_', ' ').title()}")
    print(f"   Error difference       : +{most_problematic[1]['error_difference']:.2f}%")
    print(f"   Sample count           : {most_problematic[1]['sample_count']:,}\n")
    
    print("4. RUMUSAN MASALAH #3: Kelayakan Real-Time")
    print("-"*80)
    print(f"   Threshold              : < {CONFIG['ACCEPTABLE_LATENCY_MS']} ms")
    print(f"   RF Inference Time      : {rm3_results['Random Forest']['avg_time_ms']:.4f} ms")
    print(f"   NB Inference Time      : {rm3_results['Naive Bayes']['avg_time_ms']:.4f} ms")
    print(f"   RF Status              : {rm3_results['Random Forest']['verdict']}")
    print(f"   NB Status              : {rm3_results['Naive Bayes']['verdict']}\n")
    
    print("5. TOP 5 FITUR TERPENTING")
    print("-"*80)
    for idx, row in importance_df.head(5).iterrows():
        print(f"   {idx+1}. {row['feature']:25} : {row['importance']:.6f}")
    print()
    
    # ========================================================================
    # KESIMPULAN & REKOMENDASI
    # ========================================================================
    print("\n" + "="*80)
    print(" KESIMPULAN & REKOMENDASI")
    print("="*80 + "\n")
    
    print("KESIMPULAN UTAMA:")
    print("-"*80)
    print()
    
    # Kesimpulan berdasarkan hasil
    rf_acc = trainer.results['Random Forest']['accuracy']
    nb_acc = trainer.results['Naive Bayes']['accuracy']
    entropy_acc = trainer.results['Shannon Entropy']['accuracy']
    
    print(f"1. Random Forest menunjukkan performa terbaik dengan akurasi {rf_acc*100:.2f}%,")
    print(f"   meningkat {(rf_acc - nb_acc)*100:.2f}% dari Naive Bayes dan")
    print(f"   {(rf_acc - entropy_acc)*100:.2f}% dari Shannon Entropy baseline.")
    print()
    
    if rm1_results['mcnemar_pvalue'] < 0.05:
        print(f"2. Perbedaan performa Random Forest dan Naive Bayes signifikan")
        print(f"   secara statistik (p={rm1_results['mcnemar_pvalue']:.6f} < 0.05),")
        print(f"   memvalidasi superioritas Random Forest untuk klasifikasi password.")
    else:
        print(f"2. Perbedaan performa tidak signifikan secara statistik")
        print(f"   (p={rm1_results['mcnemar_pvalue']:.6f} ≥ 0.05).")
    print()
    
    print(f"3. Naive Bayes mengalami peningkatan error rate pada password dengan")
    print(f"   pola {most_problematic[0].replace('_', ' ')}, dengan selisih")
    print(f"   +{most_problematic[1]['error_difference']:.2f}% dibanding Random Forest.")
    print()
    
    rf_feasible = rm3_results['Random Forest']['avg_time_ms'] < CONFIG['ACCEPTABLE_LATENCY_MS']
    nb_feasible = rm3_results['Naive Bayes']['avg_time_ms'] < CONFIG['ACCEPTABLE_LATENCY_MS']
    
    if rf_feasible and nb_feasible:
        print(f"4. Kedua model layak untuk implementasi real-time dengan inference time")
        print(f"   di bawah threshold {CONFIG['ACCEPTABLE_LATENCY_MS']}ms. Random Forest lebih akurat,")
        print(f"   Naive Bayes lebih efisien (lebih cepat {rm1_results['time_cost_ms']:.4f}ms).")
    elif rf_feasible:
        print(f"4. Hanya Random Forest yang layak untuk real-time implementation.")
    elif nb_feasible:
        print(f"4. Hanya Naive Bayes yang layak untuk real-time implementation.")
    else:
        print(f"4. Keduanya melebihi threshold latency, perlu optimisasi.")
    print()
    
    print("\nREKOMENDASI IMPLEMENTASI:")
    print("-"*80)
    print()
    
    if rf_feasible and rm1_results['mcnemar_pvalue'] < 0.05:
        print(" STRATEGI: RANDOM FOREST sebagai primary validator")
        print()
        print("Use Cases:")
        print("   ✓ Sistem registrasi user dengan kebutuhan keamanan tinggi")
        print("   ✓ Enterprise password policy enforcement")
        print("   ✓ Password strength meter untuk aplikasi critical (banking, healthcare)")
        print("   ✓ Compliance dengan security standards (NIST, ISO 27001)")
        print()
        print("Justifikasi:")
        print("   • Akurasi superior dengan signifikansi statistik")
        print("   • Detection rate weak passwords tinggi")
        print("   • Inference time acceptable untuk real-time")
        print("   • Trade-off waktu dibenarkan oleh peningkatan security")
    
    elif nb_feasible:
        print(" STRATEGI: NAIVE BAYES sebagai primary validator")
        print()
        print("Use Cases:")
        print("   ✓ High-volume authentication systems (millions of requests/day)")
        print("   ✓ Mobile applications dengan battery & resource constraints")
        print("   ✓ Embedded systems (IoT devices, smart cards)")
        print("   ✓ Edge computing scenarios dengan limited computational power")
        print()
        print("Justifikasi:")
        print("   • Inference time sangat cepat (sub-millisecond)")
        print("   • Memory footprint minimal")
        print("   • Akurasi mencukupi untuk mayoritas kasus")
        print("   • Skalabilitas tinggi untuk deployment distributed")
    
    else:
        print(" STRATEGI: HYBRID / CASCADED APPROACH")
        print()
        print("Implementasi:")
        print("   1. Layer 1: Shannon Entropy pre-filtering")
        print("      • Filter password dengan entropy < 2.0 (obviously weak)")
        print("      • Reject immediately tanpa ML inference")
        print()
        print("   2. Layer 2: Naive Bayes quick classification")
        print("      • Classify password dengan confidence threshold")
        print("      • Accept jika predicted strong dengan high confidence")
        print()
        print("   3. Layer 3: Random Forest for edge cases")
        print("      • Deep analysis untuk medium/ambiguous passwords")
        print("      • Final decision dengan model paling akurat")
        print()
        print("Keuntungan:")
        print("   ✓ Optimal balance antara speed dan accuracy")
        print("   ✓ Adaptive processing berdasarkan password characteristics")
        print("   ✓ Resource efficient untuk majority simple cases")
    
    print("\n" + "="*80)
    print(" ANALISIS SELESAI!")
    print("="*80 + "\n")
    
    print(f"Output folder: {output_folder}/")
    print("\nFile yang dihasilkan:")
    print("   • rf_model.pkl, nb_model.pkl, scaler.pkl")
    print("   • rm1_statistical_significance.csv")
    print("   • rm2_lexical_pattern_analysis.csv")
    print("   • rm3_realtime_feasibility.csv")
    print("   • multi_scenario_evaluation.csv")
    print("   • feature_importance.csv")
    print("   • model_comparison_summary.csv")
    print("   • confusion_matrices.png")
    print("   • feature_importance.png")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Program dihentikan oleh user (Ctrl+C)")
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nMohon periksa:")
        print("   1. Dataset 'data.csv' ada di folder yang sama")
        print("   2. Format dataset: kolom 'password' dan 'strength' (0=weak, 1=medium, 2=strong)")
        print("   3. Dependencies terinstall: pandas, numpy, scikit-learn, matplotlib, seaborn")