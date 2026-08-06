# iter-003 第5轮评估报告（588000，59 列宽表持续迭代）

日期：2026-08-06　批次：iter003_round5
前置：库 69；本轮按经验定向挖 4 个方向。

## 一句话结果

56 个候选 → eval-v2 门槛过 23 → 对 69 因子库 pooled Spearman 去重砍 6 → **入库 17，库总计 86**。

## 批次构成

| 组 | 个数 | 过筛 | 入库 |
|---|---|---|---|
| R5A_z_vs_velocity_new_states | 14 | 4 | 3 |
| R5B_velocity_construction_variants | 14 | 6 | 5 |
| R5C_conditioned_interactions | 14 | 7 | 7 |
| R5D_wide_gate_hidden_trade | 14 | 6 | 2 |

## 入库因子（按最强 OOS |t| 排序）

| 因子 | 含义 | 过几个h | 最强 OOS IC（t） | 面板\|ρ\| |
|---|---|---|---|---|
| oir_zaccel_extreme_15s | Acceleration-weighted touch-regime stret… | 5 | 30s +0.0669（+27.74） | 0.254 |
| wdi_zvel_x_spread_z | Spread-stress-gated depth-imbalance velo… | 5 | 15s -0.1484（-21.13） | 0.271 |
| oir_z_velpos_gate_15s | Build-up-isolated touch-imbalance level:… | 2 | 30s +0.1292（+18.96） | 0.644 |
| oir_zcross_x_spread_z | Spread-stress-gated touch-queue regime f… | 5 | 15s -0.1358（-15.52） | 0.343 |
| fullbook_imb_zcross_x_spread_z | Spread-stress-gated broad-book regime fl… | 5 | 30s -0.0523（-14.81） | 0.202 |
| lastgap_zvel_extreme_15s | Extremity-weighted aggressor-side veloci… | 3 | 15s +0.0350（+12.70） | 0.188 |
| oir_zcross_x_ti_sign | Trade-confirmed touch-queue flips: oir_z… | 2 | 15s +0.1153（+12.29） | 0.324 |
| oir_zcross_x_rvhi | Turbulence-isolated touch-queue flips: o… | 2 | 30s +0.0745（+12.06） | 0.425 |
| wdi_zcross_x_spread_z | Spread-stress-gated depth-imbalance regi… | 5 | 15s -0.1211（-10.52） | 0.278 |
| wdi_zvel_2sig_extreme_15s | Tail-isolated depth-imbalance velocity: … | 4 | 15s +0.0934（+7.85） | 0.205 |
| wdi_zcross_x_rvlo | Calm-isolated depth-stack rebuilds: wdi_… | 5 | 15s +0.0946（+7.41） | 0.24 |
| iopv_prem_z_cross_vel_15s | Premium-arbitrage regime reversal events… | 1 | 15s -0.0234（-4.41） | 0.316 |
| microprice_dev_zvel_2sig_extreme_15s | Tail-isolated microprice-deviation veloc… | 2 | 30s +0.0441（+2.98） | 0.151 |
| ofi_zvel_thin_gate | Regime-crossing events active ONLY under… | 2 | 15s +0.0350（+2.82） | 0.214 |
| depth5_imb_z_velneg_gate_15s | Decay-isolated visible-depth level: z_30… | 1 | 900s +0.0413（+2.40） | 0.361 |
| ltns_z_cross_vel_15s | Large-trade direction regime reversal ev… | 2 | 300s +0.0090（+2.20） | 0.274 |
| vis_hid_share_x_ofi_60s | Passive flow on a concentrated touch: th… | 1 | 900s -0.0856（-2.10） | 0.053 |

## 去重

砍 6 个：

- **lastgap_spread_wide_gate**：rho>=0.999 clone of library factor (spread multiplier ~constant on 588000 tight market -> same series); library precedence; kill |OOS t|=5.44 (1h)
- **ofi_15sz_spread_wide_gate**：rho>=0.999 clone of library factor (spread multiplier ~constant on 588000 tight market -> same series); library precedence; kill |OOS t|=5.20 (5h)
- **oir_mom_spread_wide_gate**：rho>=0.999 clone of library factor (spread multiplier ~constant on 588000 tight market -> same series); library precedence; kill |OOS t|=14.34 (2h)
- **top5div_spread_wide_gate**：rho>=0.999 clone of library factor (spread multiplier ~constant on 588000 tight market -> same series); library precedence; kill |OOS t|=14.00 (4h)
- **lastgap_z_cross_vel_15s**：keep higher |OOS t| (new dedup rule; rho in [0.85,0.999)); kill |OOS t|=9.95 (3h)
- **wdi_zaccel_extreme_15s**：keep higher |OOS t| (new dedup rule; rho in [0.85,0.999)); kill |OOS t|=13.15 (5h)

观察名单（0.70–0.85，记账不砍）：
- ofi_concord_x_spread_z x ofi_persist_x_spread_z rho=+0.826
- rv_asym_300s x mid_roll_range_pos_300s rho=+0.794
- conc_imb_z_300s x hidden_bid_support_z_300s rho=-0.792
- ofi_concord_wide_gate x ofi_persist_wide_gate rho=+0.791
- rv_asym_300s x iopv_vel_drift_300s rho=+0.784
- flow_divergence_120s x flow_divergence_300s rho=+0.783
- conc_imb_z_300s x top5_book_div_z_300s rho=+0.778
- wdi_mom_30s x depth5_delta_30s rho=+0.767
- mid_day_range_pos x dev_from_open_bps rho=+0.767
- oir_zcross_x_rvhi x oir_z_cross_vel_15s rho=+0.763
- flow_divergence_120s x flow_divergence_60s rho=+0.763
- wdi_z_cross_vel_15s x oir_z_cross_vel_15s rho=+0.740
- wdi_zvel_extreme_15s x depth5_imb_zvel_extreme_15s rho=+0.738
- div_pos_frac_300s x hidden_imb_pos_frac_300s rho=-0.733
- range_pos_x_spread_z x dev_open_x_spread_z rho=+0.728
- wdi_mom_180s x top5_book_div_z_300s rho=+0.726
- ofi_concord_15_60 x ofi_sign_persist_60s rho=+0.723
- ofi_per_depth_z_300s x iopv_vel_z_300s rho=+0.721
- ltns_z_cross_vel_15s x ti60_z_cross_vel_15s rho=+0.721
- ofi_per_depth_z_300s x slope_gated_ofi_300s rho=+0.713
- log_mid_ret_120s x mid_roll_range_pos_300s rho=+0.710
- mid_day_range_pos x div_pos_frac_300s rho=-0.703

## 死因地图

| 组/模式 | 个数 | 因子 |
|---|---|---|
| R5A/is_dead | 3 | rv60_logz_cross_vel_15s, rv300_logz_cross_vel_60s, rv300_logzvel_div_15s |
| R5A/oos_collapse | 7 | iopv_prem_zvel_extreme_60s, iopv_prem_zvel_div_60s, rv60_logzvel_extreme_15s, rv60_logzvel_div_60s, ltns_zvel_div_60s, event_int_z_cross_vel_15s, event_int_zvel_extreme_60s |
| R5B/is_dead | 7 | microprice_dev_z_accel_sign_15s, ofi_z_accel_sign_15s, wdi_vel15_x_vel60, fullbook_imb_vel15_x_vel60, ti60_vel15_x_vel60, oir_z_x_velfrac_15s, ofi_z_x_vel_15s |
| R5B/oos_collapse | 1 | fullbook_imb_z_x_vel_15s |
| R5C/is_dead | 4 | wdi_zvel_x_rv_regime, wdi_zvel_x_ofi_sign, wdi_zvel_x_oir_zvel, oir_zcross_x_fullbook_imb_zvel |
| R5C/oos_collapse | 3 | fullbook_imb_zcross_x_rv_regime, fullbook_imb_zcross_x_ti_sign, fullbook_imb_zcross_x_wdi_zvel |
| R5D/is_dead | 3 | ofi_pdepth_wide_walk, div_vis_wide_walk, hidden_ask_x_ltns_15s |
| R5D/oos_collapse | 5 | hidden_bid_x_buyvel_15s, hidden_ask_x_sellvel_15s, hidden_imb_x_ti_agree_60s, hidden_imb_x_ti_disagree_60s, hidden_bid_x_ltns_15s |

- R5A: z-velocity template does NOT generalize to volatility/premium/intensity state -- rv60/rv300/iopv_premium/event_int z-vel all IS-dead or OOS-collapse; only lastgap & ltns (gap/large-trade state) survive. Template is book/trade-imbalance + gap specific.
- R5B: raw velocity products (vel15_x_vel60), z_x_vel (non-extreme), and sign-gated accel are IS-dead -- extremeness weighting is essential; bare acceleration dies.
- R5C: stacking two book-imbalance signals (fullbook_x_wdi_zvel, wdi_x_oir_zvel, fullbook_x_oir_zvel) collapses -- one economic dimension, cross-multiply adds nothing. spread_z remains the ONLY live conditioning gate (rv_regime/ti_sign/ofi_sign die).
- R5D: _spread_wide_gate degenerates to rho>=0.999 CLONES of its base on 588000 (spread too tight/constant for the multiplier to add info) -- 4/4 killed as clones. hidden-depth x trade-velocity all OOS-collapse.

## 经验沉淀

- BREAKTHROUGH: z-ACCELERATION (2nd derivative) extremeness beats z-velocity. oir_zaccel_extreme_15s OOS t=27.74 across all 5 horizons (30s) -- strongest signal in the project. The acceleration (change-of-velocity) of a stretched book regime, extremeness-weighted, is the new primary direction: extend to fullbook_imb, depth5_imb, microprice_dev, ti60.
- z-velocity template is book/trade-imbalance + gap SPECIFIC: it dies on rv, iopv_premium, event_int state (R5A 10/14 fail). Volatility and premium levels do not carry the z-vs-velocity mismatch signal. Stop applying it to vol/premium.
- Extremeness weighting is mandatory: bare z_x_vel, vel15_x_vel60, z_accel_sign all IS-dead; only the extreme/2sigma-weighted variants live (2sig_extreme passed on wdi & microprice).
- spread_z is the ONLY live conditioning gate (3/3 zcross_x_spread_z pass); rv_regime / ti_sign / ofi_sign gates on zcross all die. spread-z gating now near saturated across the library.
- wide_gate (x spread LEVEL) is a DEAD END on 588000: 4/4 are rho>=0.999 clones of their base (spread too tight/constant). Stop multiplying by spread level.
- Cross-multiplying two book-imbalance signals collapses (rho-high, OOS-collapse): the book-imbalance family is one economic dimension.
- NEW dedup rule worked: killed the weaker twin in 2 batch pairs (lastgap z_cross_vel < zvel_extreme; wdi_zaccel < oir_zaccel t 13 vs 28) instead of first-in, and kept library on 4 rho~1.0 clones where t was identical.

## 归档

- library/candidates.json 键 `iter003_round5`；scratch-iter003/：reports_5（PASS JSON）、round5_admitted_detail.json、round5_deathmap.json、iter003_round5_corr.json/.txt、本报告。服务器副本 /data/factor_lzt/iterations/，日志 /data/factor_lzt/logs/iter003_round5.log。
