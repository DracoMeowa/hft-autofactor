# iter-003 第6轮评估报告（588000，59 列宽表持续迭代）

日期：2026-08-06　批次：iter003_round6
前置：库 86；本轮按经验定向挖 4 个方向。

## 一句话结果

56 个候选 → eval-v2 门槛过 38 → 对 86 因子库 pooled Spearman 去重砍 15 → **入库 28，库总计 114**。

## 批次构成

| 组 | 个数 | 过筛 | 入库 |
|---|---|---|---|
| R6A | 14 | 9 | 6 |
| R6B | 14 | 8 | 4 |
| R6C | 14 | 12 | 9 |
| R6D | 14 | 9 | 9 |

## 入库因子（按最强 OOS |t| 排序）

| 因子 | 含义 | 过几个h | 最强 OOS IC（t） | 面板\|ρ\| |
|---|---|---|---|---|
| oir_zaccel_ts_agree | Timescale-confirmation acceleration: the… | 5 | 15s +0.1100（+22.78） | 0.324 |
| fullbook_imb_zaccel_extreme_15s | Acceleration-weighted broad-book regime … | 5 | 900s +0.0132（+21.46） | 0.05 |
| oir_zvel_1p5sig_extreme_15s | Loose-tail-isolated touch-imbalance velo… | 4 | 15s +0.0981（+16.18） | 0.298 |
| oir_vel_accel_div_15s | Touch-rebuild overextension vs its curva… | 5 | 15s +0.1751（+15.92） | 0.469 |
| wdi_zaccel_ts_agree | Timescale-confirmation acceleration: the… | 4 | 15s +0.1150（+15.51） | 0.306 |
| oir_zjerk_extreme_15s | Jerk-weighted touch-regime stretch: the … | 5 | 30s +0.0447（+15.41） | 0.182 |
| wdi_zvel_1p5sig_extreme_15s | Loose-tail-isolated depth-imbalance velo… | 5 | 15s +0.1305（+14.33） | 0.309 |
| ofi_30s_zvel_extreme_15s | Extremity-weighted medium-window book-fl… | 5 | 900s +0.0231（+14.01） | 0.068 |
| fullbook_imb_vel_accel_div_15s | Broad-book build overextension vs curvat… | 4 | 30s +0.0939（+12.08） | 0.123 |
| wdi_zvel_neg_extreme_15s | Short-stretch-isolated depth-imbalance v… | 3 | 15s -0.0750（-10.60） | 0.154 |
| ltns_zaccel_extreme_15s | Acceleration-weighted institutional-foot… | 1 | 900s +0.0069（+8.27） | 0.025 |
| fullbook_imb_zvel_2p5sig_extreme_15s | Tail-isolated broad-book velocity: the 1… | 2 | 15s +0.0327（+7.24） | 0.112 |
| wdi_zvel_pos_extreme_15s | Long-stretch-isolated depth-imbalance ve… | 4 | 30s +0.0477（+7.18） | 0.138 |
| depth5_imb_zaccel_extreme_15s | Acceleration-weighted visible-stack regi… | 4 | 60s +0.0337（+6.98） | 0.141 |
| ofi_15s_zaccel_extreme_15s | Acceleration-weighted fast book-flow reg… | 4 | 15s +0.0330（+5.61） | 0.041 |
| wdi_zvel_3sig_extreme_15s | Deep-tail-isolated depth-imbalance veloc… | 5 | 15s +0.0315（+5.46） | 0.076 |
| fullbook_imb_zaccel_2p5sig_extreme_15s | Tail-isolated broad-book acceleration: t… | 3 | 15s +0.0232（+4.43） | 0.037 |
| ofi_15s_zvel_extreme_15s | Extremity-weighted fast book-flow veloci… | 4 | 15s +0.0604（+4.32） | 0.057 |
| ofi_60s_zvel_extreme_15s | Extremity-weighted book-flow velocity: t… | 4 | 900s +0.0181（+4.19） | 0.18 |
| ofi_60s_zaccel_extreme_15s | Acceleration-weighted book-flow regime s… | 4 | 15s +0.0368（+3.93） | 0.067 |
| microprice_dev_zaccel_2sig_extreme_15s | Tail-isolated fair-value-lead accelerati… | 3 | 300s +0.0099（+3.91） | 0.1 |
| fullbook_imb_zaccel_ts_agree | Timescale-confirmation acceleration on t… | 3 | 15s +0.0581（+3.33） | 0.13 |
| depth5_imb_zaccel_2sig_extreme_15s | Tail-isolated visible-stack acceleration… | 2 | 60s +0.0210（+3.30） | 0.076 |
| fullbook_imb_zjerk_extreme_15s | Jerk-weighted broad-book stretch: the 15… | 1 | 60s +0.0179（+2.66） | 0.026 |
| ofi_30s_zaccel_extreme_15s | Acceleration-weighted medium-window book… | 1 | 900s +0.0109（+2.54） | 0.051 |
| ti60_zaccel_2sig_extreme_15s | Tail-isolated aggressive-flow accelerati… | 1 | 900s +0.0170（+2.47） | 0.006 |
| oir_zaccel_2p5sig_extreme_15s | Tail-isolated touch-imbalance accelerati… | 1 | 300s +0.0189（+2.15） | 0.062 |
| fullbook_imb_zvel_neg_extreme_15s | Short-stretch-isolated broad-book veloci… | 1 | 15s -0.0316（-2.06） | 0.089 |

## 去重

砍 15 个：

- **microprice_dev_zaccel_extreme_15s**：keep higher |OOS t| (rho in [0.85,0.999)); kill |OOS t|=20.28 (5h)
- **lastgap_vel_accel_div_15s**：keep higher |OOS t| (rho in [0.85,0.999)); kill |OOS t|=10.20 (3h)
- **wdi_zvel_extreme_15s**：keep higher |OOS t| (rho in [0.85,0.999)); kill |OOS t|=13.87 (5h)
- **oir_zvel_extreme_decay_15s**：keep higher |OOS t| (rho in [0.85,0.999)); kill |OOS t|=14.77 (5h)
- **wdi_zvel_extreme_decay_15s**：keep higher |OOS t| (rho in [0.85,0.999)); kill |OOS t|=14.03 (5h)
- **fullbook_imb_zvel_extreme_decay_15s**：keep higher |OOS t| (rho in [0.85,0.999)); kill |OOS t|=9.77 (3h)
- **wdi_zvel_extreme_15s**：keep higher |OOS t| (rho in [0.85,0.999)); kill |OOS t|=13.87 (5h)
- **ti60_zaccel_extreme_15s**：keep higher |OOS t| (rho in [0.85,0.999)); kill |OOS t|=4.29 (1h)
- **wdi_vel_accel_div_15s**：keep higher |OOS t| (rho in [0.85,0.999)); kill |OOS t|=15.46 (5h)
- **wdi_zvel_extreme_decay_15s**：keep higher |OOS t| (rho in [0.85,0.999)); kill |OOS t|=14.03 (5h)
- **lastgap_zjerk_extreme_15s**：keep higher |OOS t| (rho in [0.85,0.999)); kill |OOS t|=3.58 (1h)
- **wdi_zvel_extreme_decay_15s**：keep higher |OOS t| (rho in [0.85,0.999)); kill |OOS t|=14.03 (5h)
- **wdi_zjerk_extreme_15s**：keep higher |OOS t| (rho in [0.85,0.999)); kill |OOS t|=10.59 (5h)
- **oir_zvel_extreme_decay_15s**：keep higher |OOS t| (rho in [0.85,0.999)); kill |OOS t|=14.77 (5h)
- **lastgap_zaccel_extreme_15s**：keep higher |OOS t| (rho in [0.85,0.999)); kill |OOS t|=4.84 (2h)

观察名单（0.70–0.85，记账不砍）：
- oir_zjerk_extreme_15s x oir_zaccel_extreme_15s rho=+0.847
- oir_vel_accel_div_15s x wdi_zvel_extreme_15s rho=+0.834
- ofi_concord_x_spread_z x ofi_persist_x_spread_z rho=+0.826
- fullbook_imb_zaccel_extreme_15s x fullbook_imb_zjerk_extreme_15s rho=+0.817
- oir_zaccel_ts_agree x wdi_zaccel_ts_agree rho=+0.815
- mid_roll_range_pos_300s x rv_asym_300s rho=+0.794
- conc_imb_z_300s x hidden_bid_support_z_300s rho=-0.792
- ofi_concord_wide_gate x ofi_persist_wide_gate rho=+0.791
- iopv_vel_drift_300s x rv_asym_300s rho=+0.784
- flow_divergence_120s x flow_divergence_300s rho=+0.783
- conc_imb_z_300s x top5_book_div_z_300s rho=+0.778
- oir_vel_accel_div_15s x oir_z_cross_vel_15s rho=+0.771
- ofi_15s_zvel_extreme_15s x ofi_15s_zaccel_extreme_15s rho=+0.770
- depth5_delta_30s x wdi_mom_30s rho=+0.767
- mid_day_range_pos x dev_from_open_bps rho=+0.767
- oir_z_cross_vel_15s x oir_zcross_x_rvhi rho=+0.763
- flow_divergence_120s x flow_divergence_60s rho=+0.763
- wdi_z_cross_vel_15s x oir_z_cross_vel_15s rho=+0.740
- wdi_zvel_extreme_15s x depth5_imb_zvel_extreme_15s rho=+0.738
- div_pos_frac_300s x hidden_imb_pos_frac_300s rho=-0.733
- wdi_zvel_neg_extreme_15s x wdi_zvel_2sig_extreme_15s rho=-0.728
- range_pos_x_spread_z x dev_open_x_spread_z rho=+0.728
- wdi_mom_180s x top5_book_div_z_300s rho=+0.726
- ofi_concord_15_60 x ofi_sign_persist_60s rho=+0.723
- ofi_per_depth_z_300s x iopv_vel_z_300s rho=+0.721
- ti60_z_cross_vel_15s x ltns_z_cross_vel_15s rho=+0.721
- oir_vel_accel_div_15s x oir_zaccel_extreme_15s rho=+0.713
- ofi_per_depth_z_300s x slope_gated_ofi_300s rho=+0.713
- log_mid_ret_120s x mid_roll_range_pos_300s rho=+0.710
- depth5_imb_zaccel_extreme_15s x depth5_imb_zvel_extreme_15s rho=+0.706
- mid_day_range_pos x div_pos_frac_300s rho=-0.703
- oir_zaccel_ts_agree x oir_zaccel_extreme_15s rho=+0.701

## 死因地图

| 组/模式 | 个数 | 因子 |
|---|---|---|
| R6A/is_dead | 3 | ti60_zaccel_x_15s, lastgap_zaccel_x_15s, ltns_zaccel_x_15s |
| R6A/oos_collapse | 2 | fullbook_imb_zaccel_x_15s, ltns_zaccel_2sig_extreme_15s |
| R6B/is_dead | 2 | fullbook_imb_zvel_x_zaccel_15s, lastgap_zvel_x_zaccel_15s |
| R6B/oos_collapse | 4 | oir_zvel_x_zaccel_15s, wdi_zvel_x_zaccel_15s, oir_zvel_zaccel_disagree_extreme_15s, wdi_zvel_zaccel_disagree_extreme_15s |
| R6C/retention_or_sign | 2 | wdi_zaccel_3sig_extreme_15s, oir_zvel_pos_extreme_15s |
| R6D/is_dead | 4 | oir_zaccel_15s_x_60s, fullbook_imb_zaccel_15s_x_60s, n_trades_60s_zvel_extreme_15s, n_trades_60s_zaccel_extreme_15s |
| R6D/oos_collapse | 1 | wdi_zaccel_15s_x_60s |

- R6A (z-accel generalization): the z-acceleration x base-level cross-multiply (`_zaccel_x_15s`) is IS-dead on ti60/lastgap/ltns and OOS-collapse on fullbook/ltns 2sig -- the acceleration template does NOT take the x-base form. It lives only as extremity-weighted (`_zaccel_extreme`, e.g. fullbook_imb t=21.46) or timescale-confirmed (`_ts_agree`). Same one-economic-dimension collapse as round-5's book-imbalance cross-multiply.
- R6B (velocity construction): co-intensification velocity x acceleration (`_zvel_x_zaccel`) collapses (fullbook/lastgap IS-dead, oir/wdi OOS-collapse); the disagreement-extreme variants OOS-collapse too. The DIVISION form (`_vel_accel_div`: z-vel over 1+|z-accel|, the exhaustion question) and the decay variants win -- velocity outpacing its own faded curvature is the live signal, not co-intensification.
- R6C (threshold sweep): the 3sig gate dies (retention fail, too tight) and the pos-only one-sided gate dies (retention); the 1.5sig gate ADMITS (oir/wdi zvel_1p5sig, t=14-16). The round-5 2sig boundary is not special -- moderate ~13%-tail stretch carries informed velocity too. Loosen, keep two-sided.
- R6D (timescale confirmation): the disagreement-carrying `_zaccel_15s_x_60s` products are IS-dead (oir/fullbook/n_trades) or OOS-collapse (wdi); their confirmation-only `_ts_agree` replacements ADMIT (oir t=22.78 strongest of round, wdi t=15.5). Multi-horizon consensus (fire only when 15s & 60s accelerations agree, disagreement exactly zero) beats carrying the slow direction through disagreement. n_trades_60s as a vel/accel substrate is IS-dead -- trade count is not informed flow on 588000.

## 经验沉淀

- BREAKTHROUGH: TIMESCALE-CONFIRMATION gate. `*_zaccel_ts_agree` fires the 15s z-acceleration amplified by the 60s z-acceleration magnitude ONLY when both point the same direction (disagreement rows exactly zero). oir_zaccel_ts_agree OOS t=22.78 across all 5 horizons -- the new project strongest. Multi-horizon consensus (institutional quoting committing at increasing speed over both seconds AND a full minute) is costlier to fake than any single-scale push. Extend ts_agree to fullbook, microprice, ofi, lastgap.
- JERK (3rd derivative) is live: `oir_zjerk_extreme_15s` OOS t=15.41 (5h). The change-rate of acceleration fires only on curvature BREAKS (steady acceleration scores ~0), complementing z-accel which fires on steady intensifying. BUT jerk is rho~0.86-0.88 with z-accel on the same base -- keep the stronger per base (z-accel won here); develop jerk where acceleration is steady but its rate is breaking.
- VELOCITY-EXHAUSTION division is a new strong form: `vel_accel_div` = z(velocity) / (1 + |z(acceleration)|). oir_vel_accel_div OOS t=15.92, IC=0.175 (highest per-row IC in the round). It asks the OPPOSITE of co-intensification: velocity outpacing its own curvature fuel = an overextended thrust about to exhaust. Beat plain zvel_extreme (killed 3x) and the decay variants. The exhaustion question is a new live axis.
- z-ACCELERATION TEMPLATE generalizes beyond oir: fullbook_imb_zaccel_extreme (t=21.46, broad-book depth beyond L5 -- passive queue/redeem tilt the L5 engines cannot see) and wdi_zaccel_ts_agree (t=15.5) both admit. Round-5's oir_zaccel was not base-specific; the acceleration-of-a-stretched-regime signal is base-portable. BUT only `_extreme` and `_ts_agree` survive -- `_x_15s` cross-multiply with the base level is dead everywhere.
- THRESHOLD SWEEP resolves: 1.5sig gate retains signal (oir/wdi zvel_1p5sig admitted, t~14-16), 3sig dies (retention fail, too tight). The round-5 2sig boundary is NOT special -- moderate ~13%-tail stretch carries informed velocity too. Pos-only one-sided gates die (retention). Rule: gates two-sided and loose (1.5sig); tight 3sig discards signal.
- VELOCITY-EXTREME template is SATURATED: plain `*_zvel_extreme_15s` killed repeatedly this round (wdi_zvel_extreme killed 3x by decay/div variants). Base extremity-weighted velocity has been fully harvested; new velocity variants must add structure (division=exhaustion, decay weighting, or timescale confirmation) to survive dedup.
- wdi ~= oir on 588000 (rho~0.86-0.90): the 5-level depth imbalance and the top-of-book imbalance are economically near-duplicate on this tight market; oir (the most actively reposted touch slot) consistently wins dedup. Stop developing wdi and oir in parallel -- treat oir as representative, wdi confirmatory only.
- NEW DEDUP RULE held: all 15 killed were 0.85-0.998 same-family duplicates kept on lower |OOS t| (e.g. microprice_dev_zaccel rho=0.998 vs oir_zaccel -- at the touch microprice_dev IS oir). No rho>=0.999 near-identical pair appeared this round, so library-precedence never triggered; the keep-higher-|t| branch decided every kill.

## 归档

- library/candidates.json 键 `iter003_round6`；scratch-iter003/：reports_6（PASS JSON）、round6_admitted_detail.json、round6_deathmap.json、iter003_round6_corr.json/.txt、本报告。服务器副本 /data/factor_lzt/iterations/，日志 /data/factor_lzt/logs/iter003_round6.log。
