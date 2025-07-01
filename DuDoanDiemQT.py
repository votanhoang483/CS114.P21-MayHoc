# Import necessary libraries
import pandas as pd
import numpy as np

try:
    # Đọc file
    df = pd.read_csv("./data/annonimized.csv")
    df_gt = pd.read_csv("./data/qt-public.csv")
    print("✅ Đã đọc dữ liệu")
    
    # Kiểm tra cấu trúc dữ liệu
    print(f"Shape của df: {df.shape}")
    print(f"Columns của df: {df.columns.tolist()}")
    print(f"Shape của df_gt: {df_gt.shape}")
    print(f"Columns của df_gt: {df_gt.columns.tolist()}")
    
except Exception as e:
    print(f"❌ Lỗi đọc dữ liệu: {e}")
    exit(1)

# Preprocess data   
# Đổi tên cột
df.columns = ["assignment_id", "problem_id", "username", "is_final", "status",
                "pre_score", "coefficient", "language_id", "created_at",
                "updated_at", "judgement"]

# Xử lý datetime với error handling
try:
    df["created_at"] = pd.to_datetime("2025-" + df["created_at"].astype(str), 
                                    format="%Y-%m-%d %H:%M:%S", errors='coerce')
    df["updated_at"] = pd.to_datetime("2025-" + df["updated_at"].astype(str), 
                                    format="%Y-%m-%d %H:%M:%S", errors='coerce')
except Exception as e:
    print(f"⚠️ Lỗi xử lý datetime: {e}")
    # Thử format khác
    df["created_at"] = pd.to_datetime(df["created_at"], errors='coerce')
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors='coerce')

# Kiểm tra missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Xử lý missing values với kiểm tra kiểu dữ liệu
if 'pre_score' in df.columns:
    df["pre_score"] = pd.to_numeric(df["pre_score"], errors='coerce')
    df["pre_score"] = df["pre_score"].fillna(0)
    
    # Kiểm tra outliers trong pre_score
    Q1 = df["pre_score"].quantile(0.25)
    Q3 = df["pre_score"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"\nOutliers in pre_score: {len(df[(df['pre_score'] < lower_bound) | (df['pre_score'] > upper_bound)])}")

# Chuyển đổi các cột categorical
if 'status' in df.columns:
    df["status"] = df["status"].astype("category")
if 'language_id' in df.columns:
    df["language_id"] = df["language_id"].astype("category")

print("\nData types after preprocessing:")
print(df.dtypes)

# Extract features
import json
def extract_verdicts(judgement_str):
    """
    Trích xuất các loại lỗi từ verdicts. Nếu verdicts rỗng => không có lỗi.
    """
    try:
        if pd.isna(judgement_str) or judgement_str is None:
            # Không có verdicts → không có lỗi
            return {
                'has_error': 0,
                'num_total_errors': 0,
                'num_wrong': 0,
                'num_time_limit': 0,
                'num_memory_limit': 0,
                'num_runtime_error': 0,
                'has_compile_error': 0,
                'has_other_error': 0
            }

        judgement = json.loads(str(judgement_str))
        verdicts = judgement.get('verdicts', {})
        
        # Nếu verdicts rỗng hoặc không có lỗi nào
        if not isinstance(verdicts, dict) or len(verdicts) == 0:
            return {
                'has_error': 0,
                'num_total_errors': 0,
                'num_wrong': 0,
                'num_time_limit': 0,
                'num_memory_limit': 0,
                'num_runtime_error': 0,
                'has_compile_error': 0,
                'has_other_error': 0
            }
            
        # Khởi tạo
        num_wrong = 0
        num_tle = 0
        num_mle = 0
        num_rte = 0
        has_compile_error = 0
        other_errors = 0
        
        for verdict, value in verdicts.items():
            verdict_lower = verdict.lower().strip()

            # Trường hợp đặc biệt: key rỗng và value là thông báo lỗi → compile error
            if verdict_lower == "" and isinstance(value, str) and len(value.strip()) > 0:
                has_compile_error = 1
                continue

            # Cố gắng ép sang int nếu có thể
            try:
                count = int(value)
                if count < 0: count = 0
            except:
                count = 1  # Không ép được → xem là xuất hiện ít nhất 1 lần

            if 'wrong' in verdict_lower:
                num_wrong += count
            elif 'time limit' in verdict_lower:
                num_tle += count
            elif 'memory limit' in verdict_lower:
                num_mle += count
            elif 'runtime error' in verdict_lower:
                num_rte += count
            elif 'compile' in verdict_lower:
                has_compile_error = 1
            else:
                other_errors += count

        total_errors = num_wrong + num_tle + num_mle + num_rte + has_compile_error + other_errors
        has_error = 1 if total_errors > 0 else 0

        return {
            'has_error': has_error,
            'num_total_errors': total_errors,
            'num_wrong': num_wrong,
            'num_time_limit': num_tle,
            'num_memory_limit': num_mle,
            'num_runtime_error': num_rte,
            'has_compile_error': has_compile_error,
            'has_other_error': 1 if other_errors > 0 else 0
        }

    except Exception as e:
        print(f"Lỗi: {e}")
        return {
            'has_error': 0,
            'num_total_errors': 0,
            'num_wrong': 0,
            'num_time_limit': 0,
            'num_memory_limit': 0,
            'num_runtime_error': 0,
            'has_compile_error': 0,
            'has_other_error': 0
        }
        
        
# Áp dụng hàm extract_verdicts vào cột judgement
if 'judgement' in df.columns:
    print("Đang xử lý judgement...")
    judgement_features = df['judgement'].apply(extract_verdicts)
    
    # Chuyển đổi kết quả thành DataFrame
    judgement_df = pd.DataFrame(judgement_features.tolist())
    
    # Xóa các cột cũ nếu đã tồn tại
    columns_to_drop = ['has_error', 'num_total_errors', 'num_wrong',
                       'num_time_limit', 'num_memory_limit', 'num_runtime_error',
                       'has_compile_error', 'has_other_error']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Kết hợp với DataFrame gốc
    df = pd.concat([df, judgement_df], axis=1)
    
    print("✅ Đã xử lý xong judgement features")
    print(f"Shape sau khi thêm features: {df.shape}")
else:
    print("⚠️ Không tìm thấy cột 'judgement'")
    # Tạo các cột mặc định
    df['has_error'] = 0
    df['num_total_errors'] = 0
    df['num_wrong'] = 0
    df['num_time_limit'] = 0
    df['num_memory_limit'] = 0
    df['num_runtime_error'] = 0
    df['has_compile_error'] = 0
    df['has_other_error'] = 0
    
# Combine features
# Tổng hợp theo sinh viên với error handling
try:
    print("Đang tổng hợp dữ liệu theo sinh viên...")
    
    # Kiểm tra các cột cần thiết
    required_cols = ['username', 'assignment_id', 'pre_score', 'is_final']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Thiếu các cột: {missing_cols}")
        exit(1)
    
    # Tổng số bài duy nhất đã làm
    num_assignments = df.groupby('username')['assignment_id'].nunique().rename('num_assignments')
    
    # Tổng số lượt nộp
    total_submissions = df.groupby('username').size().rename('total_submissions')
    
     # Số bài có ít nhất 1 lần đúng
    correct_df = df[df['is_final'] == 1].groupby('username')['assignment_id'].nunique().rename('num_correct')
    
    # Điểm trung bình và cao nhất
    score_stats = df.groupby('username')['pre_score'].agg(['mean', 'max']).rename(columns={
        'mean': 'avg_score',
        'max': 'max_score'
    })
    
    # Tổng các loại lỗi
    error_cols = ['num_wrong', 'num_time_limit', 'num_memory_limit', 'num_runtime_error',
                  'has_compile_error', 'has_other_error']
    error_agg = df.groupby('username')[error_cols].sum()

    # Ngày nộp đầu tiên và cuối cùng (nếu có)
    if 'created_at' in df.columns and not df['created_at'].isna().all():
        time_stats = df.groupby('username')['created_at'].agg(['min', 'max']).rename(columns={
            'min': 'first_submission',
            'max': 'last_submission'
        })
    else:
        time_stats = pd.DataFrame()

    # Gộp tất cả lại
    student_features = pd.concat([
        num_assignments,
        total_submissions,
        correct_df,
        score_stats,
        error_agg,
        time_stats
    ], axis=1).fillna(0)

    # Thêm đặc trưng tỷ lệ đúng
    student_features['correct_ratio'] = student_features['num_correct'] / student_features['num_assignments']
    student_features['correct_ratio'] = student_features['correct_ratio'].fillna(0)

    # Thêm đặc trưng số ngày và tốc độ nộp bài (nếu có thời gian)
    if 'first_submission' in student_features.columns and 'last_submission' in student_features.columns:
        student_features['total_days'] = (student_features['last_submission'] - student_features['first_submission']).dt.days
        student_features['total_days'] = student_features['total_days'].replace(0, 1).fillna(1)
        student_features['submission_rate'] = student_features['total_submissions'] / student_features['total_days']

    # Reset index
    student_features = student_features.reset_index()

    print(f"✅ Đã tổng hợp xong. Shape: {student_features.shape}")

except Exception as e:
    print(f"❌ Lỗi khi tổng hợp: {e}")
    exit(1)

# Label ground truth
try:
    print("Đang gắn nhãn ground truth...")
    
    # Kiểm tra cấu trúc df_gt
    if 'hash' not in df_gt.columns:
        print("❌ Không tìm thấy cột 'hash' trong file ground truth")
        print(f"Các cột có sẵn: {df_gt.columns.tolist()}")
        exit(1)
    
    if 'diemqt' not in df_gt.columns:
        print("❌ Không tìm thấy cột 'diemqt' trong file ground truth")
        print(f"Các cột có sẵn: {df_gt.columns.tolist()}")
        exit(1)
    
    student_features = student_features.merge(df_gt, how="left", left_on="username", right_on="hash")
    
    # Phân chia dữ liệu
    unknown_students = set(df["username"]) - set(df_gt["hash"])
    test_data = student_features[student_features["username"].isin(unknown_students)]
    train_data = student_features[student_features["username"].isin(df_gt["hash"])]
    
    print(f"Số sinh viên training: {len(train_data)}")
    print(f"Số sinh viên testing: {len(test_data)}")
    
    if len(train_data) == 0:
        print("❌ Không có dữ liệu training!")
        exit(1)
    
    # Chuẩn bị features cho training
    feature_columns = [col for col in student_features.columns 
                      if col not in ["username", "hash", "diemqt", "first_submission", "last_submission"]]
    
    X_train = train_data[feature_columns]
    y_train = pd.to_numeric(train_data["diemqt"], errors="coerce")
    
    # Loại bỏ các giá trị NaN
    valid_indices = ~(y_train.isna() | X_train.isna().any(axis=1))
    X_train = X_train[valid_indices]
    y_train = y_train[valid_indices]
    
    print(f"Số mẫu training sau khi làm sạch: {len(X_train)}")
    print(f"Số features: {len(feature_columns)}")
    print(f"Features: {feature_columns}")
    
    if len(X_train) == 0:
        print("❌ Không có dữ liệu training hợp lệ!")
        exit(1)
    
except Exception as e:
    print(f"❌ Lỗi trong quá trình chuẩn bị dữ liệu: {e}")
    exit(1)
    
from sklearn.model_selection import train_test_split
# Chia train/validation split
try:
    print("Đang chia dữ liệu train/validation...")
    
    # Chia 90% train, 10% validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, 
        test_size=0.1, 
        random_state=42,
        stratify=None  # Có thể thêm stratify nếu cần
    )
    
    print(f"Tập training: {len(X_train_split)} samples")
    print(f"Tập validation: {len(X_val_split)} samples")
    
except Exception as e:
    print(f"❌ Lỗi khi chia dữ liệu: {e}")
    # Fallback: sử dụng toàn bộ dữ liệu cho training
    X_train_split, X_val_split = X_train, None
    y_train_split, y_val_split = y_train, None
X_train_split.to_csv("./data/X_train_split.csv", index=False)

# Train model and predict
# Training model
try:
    print("Đang training model...")
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # from sklearn.linear_model import LinearRegression

    # model = LinearRegression()
    # model.fit(X_train_split, y_train_split)
    
    # Import và train Random Forest

    # model = RandomForestRegressor(
    #     n_estimators=300,
    #     random_state=42,
    #     n_jobs=-1
    # )
    # model.fit(X_train_split, y_train_split)
    
    from lightgbm import LGBMRegressor
    model = LGBMRegressor(
        n_estimators=300, 
        learning_rate=0.01, 
        random_state=42,
        verbose=-1,  # Tắt log
        early_stopping_rounds=50 if X_val_split is not None else None
    )
    # Fit với validation set nếu có
    if X_val_split is not None:
        model.fit(
            X_train_split, y_train_split,
            eval_set=[(X_val_split, y_val_split)],
            eval_metric='rmse',
        )
    else:
        model.fit(X_train_split, y_train_split)
        
    print("✅ Training hoàn thành")
        
# Đánh giá trên tập training
    train_preds = model.predict(X_train_split)
    train_r2 = r2_score(y_train_split, train_preds)
    train_rmse = np.sqrt(mean_squared_error(y_train_split, train_preds))
    train_mae = mean_absolute_error(y_train_split, train_preds)
    
    print(f"\n📊 Kết quả trên tập TRAINING:")
    print(f"R² score: {train_r2:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"MAE: {train_mae:.4f}")
    
    # Đánh giá trên tập validation nếu có
    if X_val_split is not None:
        val_preds = model.predict(X_val_split)
        val_r2 = r2_score(y_val_split, val_preds)
        val_rmse = np.sqrt(mean_squared_error(y_val_split, val_preds))
        val_mae = mean_absolute_error(y_val_split, val_preds)
        
        print(f"\n📊 Kết quả trên tập VALIDATION:")
        print(f"R² score: {val_r2:.4f}")
        print(f"RMSE: {val_rmse:.4f}")
        print(f"MAE: {val_mae:.4f}")
        
        # Kiểm tra overfitting
        print(f"\n🔍 Phân tích Overfitting:")
        print(f"Gap R² (train - val): {train_r2 - val_r2:.4f}")
        print(f"Gap RMSE (val - train): {val_rmse - train_rmse:.4f}")
        
        if abs(train_r2 - val_r2) > 0.1:
            print("⚠️ Có dấu hiệu overfitting (gap R² > 0.1)")
        else:
            print("✅ Model có vẻ ổn định (không overfitting nghiêm trọng)")
    else:
        print("⚠️ Không có tập validation để đánh giá")
    
    # Dự đoán cho tập test
    if len(test_data) > 0:
        X_test = test_data[feature_columns]
        # Xử lý missing values trong test set
        X_test = X_test.fillna(X_train.mean())
        
        test_preds = model.predict(X_test)
        
        print(f"✅ Dự đoán hoàn thành cho {len(test_preds)} sinh viên")
        print(f"Mean prediction: {np.mean(test_preds):.4f}")
        print(f"Std prediction: {np.std(test_preds):.4f}")
        
        # Hiển thị feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 important features:")
        print(feature_importance.head(10))
        
        # Tạo DataFrame kết quả dự đoán
        results_df = pd.DataFrame({
            'username': test_data['username'].values,
            'predicted_score': test_preds
        })
        
        # Làm tròn điểm dự đoán đến 2 chữ số thập phân
        results_df['predicted_score'] = results_df['predicted_score'].round(2)
        
        # Sắp xếp theo username
        results_df = results_df.sort_values('username')
        
        # Xuất ra file CSV
        output_file = 'predicted_scores_QT.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"\n✅ Đã xuất kết quả dự đoán ra file: {output_file}")
        print(f"Số sinh viên được dự đoán: {len(results_df)}")
        print("\n📊 Thống kê kết quả dự đoán:")
        print(f"Điểm trung bình: {results_df['predicted_score'].mean():.2f}")
        print(f"Điểm cao nhất: {results_df['predicted_score'].max():.2f}")
        print(f"Điểm thấp nhất: {results_df['predicted_score'].min():.2f}")
        print(f"Độ lệch chuẩn: {results_df['predicted_score'].std():.2f}")
        
        print("\n🔍 Preview kết quả (10 dòng đầu):")
        print(results_df.head(10))
        
    else:
        print("⚠️ Không có dữ liệu test để dự đoán")
        
except Exception as e:
    print(f"❌ Lỗi trong quá trình training: {e}")
    import traceback
    traceback.print_exc()

