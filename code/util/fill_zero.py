import pandas as pd

def fill_zero(input_file, output_file):
    # 確保使用函數參數，而不是硬編碼
    # 假設 test_id 範圍從 0 到 18635（根據上下文調整）
    min_test_id = 0
    max_test_id = 18635

    # 讀取現有數據
    df = pd.read_csv(input_file)

    # 獲取現有 test_id 列表
    existing_ids = df['test_id'].tolist()

    # 創建完整的 test_id 範圍
    all_ids = list(range(min_test_id, max_test_id + 1))

    # 找到缺失的 test_id
    missing_ids = [tid for tid in all_ids if tid not in existing_ids]

    # 創建缺失的 test_id 數據（全 0）
    missing_data = {
        'test_id': missing_ids,
        'adult_males': [0] * len(missing_ids),
        'subadult_males': [0] * len(missing_ids),
        'adult_females': [0] * len(missing_ids),
        'juveniles': [0] * len(missing_ids),
        'pups': [0] * len(missing_ids)
    }
    missing_df = pd.DataFrame(missing_data)

    # 合併現有數據和補充數據
    combined_df = pd.concat([df, missing_df], ignore_index=True)

    # 按 test_id 排序
    combined_df = combined_df.sort_values('test_id').reset_index(drop=True)

    # 保存結果，確保不產生多餘逗號
    combined_df.to_csv(output_file, index=False, lineterminator='\n')
    print(f"已將缺失的 test_id 補 0，結果儲存至 {output_file}")

def post_process(input_file, output_file):
    # 讀取數據
    df = pd.read_csv(input_file)

    # 計算調整後的 juveniles 和 adult_females
    df['juveniles'] = (df['juveniles'] * 1.5).astype(int)  # 將 juveniles 乘以 1.5，取整數
    increase = df['juveniles'] - (df['juveniles'] / 1.5).astype(int)  # 增加的數量
    df['adult_females'] = (df['adult_females'] - increase).clip(lower=0)  # 從 adult_females 扣除，確保不為負

    # 將 pups 增加 20%
    df['pups'] = (df['pups'] * 1.2).astype(int)  # 增加 20%，取整數

    # 保存結果
    df.to_csv(output_file, index=False)
    print(f"已調整數據，結果儲存至 {output_file}")

if __name__ == "__main__":
    # fill_zero(input_file='submission_65_copy.csv', output_file='submission_65_with_zeros.csv')
    post_process(input_file='submission_65_with_zeros.csv', output_file='submission_65_with_zeros_post_processed.csv') 
