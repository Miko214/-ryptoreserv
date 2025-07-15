import re
import pandas as pd

LOG_FILE   = 'bot_log.log'
OUTPUT_CSV = 'labeled_trades.csv'

# 1. Паттерны для открытия и закрытия
pattern_open = re.compile(
    r".*\[(?P<symbol>[^\]]+)\].*ОТКРЫТА\s+(?P<side>LONG|SHORT)\s+СДЕЛКА.*?@\s*(?P<price>\d+\.\d+)"
)
pattern_tp = re.compile(
    r".*\[(?P<symbol>[^\]]+)\].*Take Profit\s*(?P<tp>\d+)\s+достигнут.*?@\s*(?P<price>\d+\.\d+)"
)
pattern_sl = re.compile(
    r".*\[(?P<symbol>[^\]]+)\].*СДЕЛКА ЗАКРЫТА по Stop Loss.*?@\s*(?P<price>\d+\.\d+)"
)

opens  = []
closes = []

with open(LOG_FILE, encoding='utf-8') as f:
    for line in f:
        timestamp = line.split(" - ")[0]

        m = pattern_open.search(line)
        if m:
            opens.append({
                'time':        timestamp,
                'symbol':      m.group('symbol'),
                'side':        m.group('side'),
                'entry_price': float(m.group('price'))
            })
            continue

        m = pattern_tp.search(line)
        if m:
            closes.append({
                'time':       timestamp,
                'symbol':     m.group('symbol'),
                'reason':     f"TP{m.group('tp')}",
                'exit_price': float(m.group('price'))
            })
            continue

        m = pattern_sl.search(line)
        if m:
            closes.append({
                'time':       timestamp,
                'symbol':     m.group('symbol'),
                'reason':     "SL",
                'exit_price': float(m.group('price'))
            })

# 2. Сопоставляем открытия и первые закрытия по времени и символу
records = []
for o in opens:
    cands = [c for c in closes if c['symbol']==o['symbol'] and c['time']>=o['time']]
    if not cands:
        continue
    c = cands[0]
    pnl = (c['exit_price'] - o['entry_price']) / o['entry_price']
    if o['side'] == 'SHORT':
        pnl = -pnl
    records.append({
        'symbol':      o['symbol'],
        'side':        o['side'],
        'entry_time':  o['time'],
        'entry_price': o['entry_price'],
        'exit_time':   c['time'],
        'exit_price':  c['exit_price'],
        'reason':      c['reason'],
        'pnl':         pnl,
        'label':       1 if pnl > 0 else 0
    })

# 3. Сохраняем в CSV
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False, float_format='%.6f')
print(f"Готово: {len(df)} сделок размечено -> {OUTPUT_CSV}")
