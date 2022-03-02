from trader import *
import json
import logging

# with open(ROOT_PATH + "/papertrading/static/stable.json") as f:
#     STOCKS = json.load(f)

with open(ROOT_PATH + "/papertrading/static/tick_instrument.json", "r") as f:
        ticks = json.load(f)
        STOCKS = ticks["equity"]

def rscore_screener(vz):
    rscore = []
    for s in vz:
        rs, rsmax, rsmin = vz[s]["rscore"], vz[s]["rscore-max"], vz[s]["rscore-min"]
        rscore.append([
            s,
            rs,
            rsmin,
            rsmax,
            rsmax - rs,
            rs - rsmin,
            vz[s]["uext"],
            vz[s]["dext"],
            vz[s]["otype"],
            vz[s]["orange"],
            vz[s]["margin_up"],
            vz[s]["margin_down"],
            vz[s]["tpo-ratio"]
        ])
    return rscore

def screener_data(end=None):
    if end is None:
        end = str(datetime.now().date())
    df = fetch_data(STOCKS, f"{end} 09:00:00", f"{end} 16:00:00", span="30minute")
    vz = {}

    for s in STOCKS:
        try:
            adf = df[df["tradingsymbol"] == s].reset_index(drop=True)
            if adf.shape[0] == 0:
                continue
            values_zones, mpm, xticks, yticks, ymax, ymin, BASE = calculate_value_zones(adf)
            vz[s] = values_zones[-1]
        except Exception as e:
            logging.error(s)
            logging.exception(e)
    rscore = rscore_screener(vz)
    return {"rscore": rscore}
