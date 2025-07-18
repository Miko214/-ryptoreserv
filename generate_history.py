import json
import ccxt
import pandas as pd
import pandas_ta as ta

# --- Параметры ---
CONFIG_FILE       = 'config.json'
OUTPUT_FILE       = 'full_history_with_indicators.csv'
HISTORY_LIMIT     = 500

# Настройки индикаторов
ATR_LENGTH        = 14
RSI_LENGTH        = 14
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9
BB_LENGTH         = 20
BB_STD            = 2.0
EMA_SHORT         = 50
EMA_LONG          = 200
VOLUME_EMA_LENGTH = 20
KAMA_LENGTH       = 10
KAMA_FAST_EMA     = 2
KAMA_SLOW_EMA     = 30
CMF_LENGTH        = 20
RVI_LENGTH        = 14

# 1) Загрузить конфиг
with open(CONFIG_FILE, encoding='utf-8') as f:
    cfg = json.load(f)
SYMBOLS   = cfg['symbols']
TIMEFRAME = cfg['timeframe']

# 2) Инициализировать биржу
exchange = ccxt.binanceusdm({'enableRateLimit': True})

def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    data = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=HISTORY_LIMIT)
    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) ATR
    df[f'ATR_{ATR_LENGTH}'] = ta.atr(
        high=df['high'], low=df['low'], close=df['close'],
        length=ATR_LENGTH
    )

    # 2) EMA ценовые
    df[f'EMA_{EMA_SHORT}'] = ta.ema(close=df['close'], length=EMA_SHORT)
    df[f'EMA_{EMA_LONG}']  = ta.ema(close=df['close'], length=EMA_LONG)

    # 3) RSI
    df[f'RSI_{RSI_LENGTH}'] = ta.rsi(close=df['close'], length=RSI_LENGTH)

    # 4) MACD
    macd_df = ta.macd(
        close=df['close'],
        fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL
    )
    df = pd.concat([df, macd_df], axis=1)
    # колонки macd_df: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9

    # 5) Bollinger Bands
    bb_df = ta.bbands(
        high=df['high'], low=df['low'], close=df['close'],
        length=BB_LENGTH, std=BB_STD
    )
    # bb_df.columns: ['BBU_20_2.0','BBM_20_2.0','BBL_20_2.0']
    df['BB_upper']  = bb_df.iloc[:, 0]
    df['BB_middle'] = bb_df.iloc[:, 1]
    df['BB_lower']  = bb_df.iloc[:, 2]
    df['BB_width']  = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

    # 6) EMA объёма
    df['VOLUME_EMA'] = ta.ema(close=df['volume'], length=VOLUME_EMA_LENGTH)

    # 7) KAMA, CMF, RVI
    df[f'KAMA_{KAMA_LENGTH}_{KAMA_FAST_EMA}_{KAMA_SLOW_EMA}'] = ta.kama(
        close=df['close'], length=KAMA_LENGTH,
        fast=KAMA_FAST_EMA, slow=KAMA_SLOW_EMA
    )
    df[f'CMF_{CMF_LENGTH}'] = ta.cmf(
        high=df['high'], low=df['low'],
        close=df['close'], volume=df['volume'],
        length=CMF_LENGTH
    )
    df[f'RVI_{RVI_LENGTH}'] = ta.rvi(
        close=df['close'], length=RVI_LENGTH
    )

    # 8) Очищаем только первые неполные строки
    required = [
        f'ATR_{ATR_LENGTH}',
        f'EMA_{EMA_SHORT}', f'EMA_{EMA_LONG}',
        f'RSI_{RSI_LENGTH}',
        f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}',
        f'MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}',
        'BB_upper', 'BB_middle', 'BB_width',
        'VOLUME_EMA',
        f'KAMA_{KAMA_LENGTH}_{KAMA_FAST_EMA}_{KAMA_SLOW_EMA}',
        f'CMF_{CMF_LENGTH}',
        f'RVI_{RVI_LENGTH}'
    ]
    df.dropna(subset=required, inplace=True)

    return df

def main():
    all_dfs = []

    for sym in SYMBOLS:
        print(f"→ Fetching {sym}")
        df = fetch_ohlcv(sym)
        if df.empty:
            print("   skipped — no data")
            continue

        df_ind = compute_all_indicators(df)
        if df_ind.empty:
            print("   skipped — indicators not ready")
            continue

        df_ind['symbol'] = sym
        all_dfs.append(df_ind)

    if not all_dfs:
        raise RuntimeError("No data to save — check fetch or indicator calc")

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv(OUTPUT_FILE, index=False, float_format='%.6f')
    print(f"✔ Saved {len(full_df)} rows → {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
