# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import r2_score
import json

try:
    # Đọc file
    df = pd.read_csv("./data/annonimized.csv")
    df_gt = pd.read_csv("./data/ck-public.csv")
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
def extract_verdicts(judgement_str):
    """
    Trích xuất thông tin verdict từ chuỗi JSON
    """
    try:
        # Kiểm tra nếu là NaN hoặc None
        if pd.isna(judgement_str) or judgement_str is None:
            return {
                'num_verdicts': 0,
                'num_wrong': 0,
                'num_time_limit': 0,
                'num_memory_limit': 0,
                'num_runtime_error': 0,
                'error_ratio': 0
            }
        
        # Parse JSON string
        judgement = json.loads(str(judgement_str))
        
        # Khởi tạo biến đếm
        total_verdicts = 0
        wrong_count = 0
        time_limit_count = 0
        memory_limit_count = 0
        runtime_error_count = 0
        
        # Get verdicts from the correct structure
        if isinstance(judgement, dict) and 'verdicts' in judgement:
            verdicts = judgement['verdicts']
            if isinstance(verdicts, dict):
                # Đếm từng loại verdict
                wrong_count = verdicts.get('WRONG', 0)
                time_limit_count = verdicts.get('Time Limit Exceeded', 0)
                memory_limit_count = verdicts.get('Memory Limit Exceeded', 0)
                runtime_error_count = verdicts.get('Runtime Error', 0)
                total_verdicts = sum(verdicts.values())
            elif isinstance(verdicts, list):
                # Nếu verdicts là list, đếm từng loại
                for verdict in verdicts:
                    if verdict == 'WRONG':
                        wrong_count += 1
                    elif verdict == 'Time Limit Exceeded':
                        time_limit_count += 1
                    elif verdict == 'Memory Limit Exceeded':
                        memory_limit_count += 1
                    elif verdict == 'Runtime Error':
                        runtime_error_count += 1
                total_verdicts = len(verdicts)
        
        # Tính tỉ lệ lỗi
        error_ratio = (wrong_count + time_limit_count + memory_limit_count + runtime_error_count) / total_verdicts if total_verdicts > 0 else 0
        
        return {
            'num_verdicts': total_verdicts,
            'num_wrong': wrong_count,
            'num_time_limit': time_limit_count,
            'num_memory_limit': memory_limit_count,
            'num_runtime_error': runtime_error_count,
            'error_ratio': error_ratio
        }
        
    except json.JSONDecodeError as e:
        print(f"Lỗi JSON: {e}")
        return {
            'num_verdicts': 0,
            'num_wrong': 0,
            'num_time_limit': 0,
            'num_memory_limit': 0,
            'num_runtime_error': 0,
            'error_ratio': 0
        }
    except Exception as e:
        print(f"Lỗi không xác định: {e}")
        return {
            'num_verdicts': 0,
            'num_wrong': 0,
            'num_time_limit': 0,
            'num_memory_limit': 0,
            'num_runtime_error': 0,
            'error_ratio': 0
        }

# Áp dụng hàm extract_verdicts vào cột judgement
if 'judgement' in df.columns:
    print("Đang xử lý judgement...")
    judgement_features = df['judgement'].apply(extract_verdicts)
    
    # Chuyển đổi kết quả thành DataFrame
    judgement_df = pd.DataFrame(judgement_features.tolist())
    
    # Xóa các cột cũ nếu đã tồn tại
    columns_to_drop = ['num_verdicts', 'num_wrong', 'num_time_limit', 'num_memory_limit', 'num_runtime_error', 'error_ratio']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Kết hợp với DataFrame gốc
    df = pd.concat([df, judgement_df], axis=1)
    
    print("✅ Đã xử lý xong judgement features")
    print(f"Shape sau khi thêm features: {df.shape}")
else:
    print("⚠️ Không tìm thấy cột 'judgement'")
    # Tạo các cột mặc định
    df['num_verdicts'] = 0
    df['num_wrong'] = 0
    df['num_time_limit'] = 0
    df['num_memory_limit'] = 0
    df['num_runtime_error'] = 0
    df['error_ratio'] = 0
    
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
    
    agg_dict = {
        'assignment_id': 'count',  # Số lượng bài tập đã làm
        'pre_score': ['mean', 'max'],  # Điểm trung bình và cao nhất
        'num_verdicts': 'mean',  # Số lần nộp bài trung bình
        'num_wrong': 'mean',  # Số lần sai trung bình
        'num_time_limit': 'mean',  # Số lần vượt thời gian trung bình
        'num_memory_limit': 'mean',  # Số lần vượt bộ nhớ trung bình
        'num_runtime_error': 'mean',  # Số lần lỗi runtime trung bình
        'error_ratio': 'mean',  # Tỷ lệ lỗi trung bình
        'is_final': 'sum'  # Số lượng bài làm đúng
    }
    
    # Thêm created_at nếu có
    if 'created_at' in df.columns and not df['created_at'].isna().all():
        agg_dict['created_at'] = ['min', 'max']
    
    student_features = df.groupby("username").agg(agg_dict).reset_index()
    
    # Flatten column names
    new_columns = ['username']
    for col in student_features.columns[1:]:
        if isinstance(col, tuple):
            new_columns.append(f"{col[0]}_{col[1]}")
        else:
            new_columns.append(col)
    
    student_features.columns = new_columns
    
    # Đổi tên cột cho dễ hiểu
    rename_dict = {
        'assignment_id_count': 'total_assignments',
        'pre_score_mean': 'avg_score',
        'pre_score_max': 'max_score',
        'num_verdicts_mean': 'avg_submissions',
        'num_wrong_mean': 'avg_wrong_attempts',
        'num_time_limit_mean': 'avg_time_limit_errors',
        'num_memory_limit_mean': 'avg_memory_limit_errors',
        'num_runtime_error_mean': 'avg_runtime_errors',
        'error_ratio_mean': 'avg_error_ratio',
        'is_final_sum': 'total_correct'
    }
    
    # Thêm datetime columns nếu có
    if 'created_at_min' in student_features.columns:
        rename_dict.update({
            'created_at_min': 'first_submission',
            'created_at_max': 'last_submission'
        })
    
    student_features = student_features.rename(columns=rename_dict)
    
    # Tạo các features bổ sung
    student_features['correct_ratio'] = student_features['total_correct'] / student_features['total_assignments']
    
    # Xử lý thời gian nếu có
    if 'first_submission' in student_features.columns and 'last_submission' in student_features.columns:
        student_features['total_days'] = (student_features['last_submission'] - student_features['first_submission']).dt.days
        student_features['total_days'] = student_features['total_days'].fillna(1)
        student_features['submission_rate'] = student_features['total_assignments'] / student_features['total_days'].replace(0, 1)
    
    print(f"✅ Tổng hợp xong. Shape: {student_features.shape}")
    
except Exception as e:
    print(f"❌ Lỗi trong quá trình tổng hợp: {e}")
    exit(1)

# Label ground truth
try:
    print("Đang gắn nhãn ground truth...")
    
    # Kiểm tra cấu trúc df_gt
    if 'hash' not in df_gt.columns:
        print("❌ Không tìm thấy cột 'hash' trong file ground truth")
        print(f"Các cột có sẵn: {df_gt.columns.tolist()}")
        exit(1)
    
    if 'CK' not in df_gt.columns:
        print("❌ Không tìm thấy cột 'CK' trong file ground truth")
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
                      if col not in ["username", "hash", "CK", "first_submission", "last_submission"]]
    
    X_train = train_data[feature_columns]
    y_train = pd.to_numeric(train_data["CK"], errors="coerce")
    
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
    
# Train model and predict
# Training model
try:
    print("Đang training model...")
    
    # Import và train LightGBM
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    model = LGBMRegressor(
        n_estimators=300, 
        learning_rate=0.05, 
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
        output_file = 'predicted_scores_CK.csv'
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

