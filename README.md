
Sports Prediction Models
Sean Ryan | Quantitative Trader & Sports Modeler
A collection of quantitative sports prediction models built and actively traded since 2014. Each model is the product of years of research into predictive statistics, market dynamics, and iterative improvement based on real trading results.

Modeling Philosophy
My approach starts with a simple but important distinction: descriptive statistics are not always predictive. A statistic that accurately describes past performance does not automatically translate into a feature that predicts future outcomes. Every statistic I use must pass two tests:

Quantitative — does this skill show consistent predictive value over time in historical data?
Logical — does the relationship make fundamental sense in the context of how the game is actually played?

This discipline is what separates a model that captures real edge from one that fits historical data but breaks down in deployment.
My research process involves sourcing advanced statistics from data providers, rigorously quality checking for accuracy and consistency, and then analyzing how each statistic can be used to predict game results. I combine this quantitative foundation with a deep understanding of how games and markets actually flow — including how lines move historically, what those movements signal, and where pricing inefficiencies exist across platforms.
All models output actionable lines across spreads, moneylines, totals, and props, going beyond win probability to produce full-market pricing.

Models & Results
NCAA Football
Monte Carlo simulation model
Two seasons of live trading results, with meaningful improvement between years as the model was refined based on post-result analysis.
SeasonROI Per TradeCLV2024ProfitablePositive202510%2.2%
2024 Season Results
![NCAAF 2024 Results](NCAAF Results 2024-2025.png)
2025 Season Results
![NCAAF 2025 Results](NCAAF Results 2025-2026.png)

NHL
Poisson distribution-based model
SeasonROI Per TradeCLV2025-20267.2%1.2%
![NHL Results](NHL Results 2025-2026.png)

PGA Golf
AI-assisted prediction model built using Claude
PeriodROIPast 2 months~7%
![PGA Results](Golf Results 2026.png)

Overall Portfolio
30%+ ROI across all prediction market activity in 2025
A consistent edge across multiple sports and bet types, validated by restrictions at more than ten domestic and international sportsbooks.

Technical Stack
ComponentDetailsLanguagePython (primary)Librariespandas, numpy, scipy, requests, BeautifulSoup, SeleniumModelingMonte Carlo simulation, Poisson distribution modeling, statistical optimizationDataWeb scraping, REST API integration, large-dataset analysisAI ToolsClaude (strategy development and model assistance)

MLB Simulator
The MLB_2026_labeled.py file in this repo is a Monte Carlo game simulator for MLB matchups. For each game it:

Loads pitcher and batter statistical rates from CSV data files
Applies park factors, weather adjustments, and situational multipliers
Simulates each at-bat probabilistically — determining outcome (K, BB, GB, LD, FB), trajectory, direction, magnitude, and baserunning results
Runs 1,000 simulations per matchup and outputs a full box score distribution

The model incorporates:

Trajectory modeling — ground balls, line drives, fly balls weighted by pitcher and batter tendencies
Directional park factors — separate factors for pull, center, and oppo directions in each ballpark
Weather adjustments — temperature and wind effects on hit rates derived empirically from historical data
Situational adjustments — base/out state multipliers for strikeout and walk rates
Starter fatigue — pitcher effectiveness decay modeled after inning 4
Extra innings rule — automatic runner on second from inning 10 onward


About
I have been building quantitative sports models and actively trading prediction markets since 2014. My professional background spans a decade in systematic strategy design at S&P Dow Jones Indices and Nasdaq, where I built trading platforms, managed 100+ systematic strategies, and led a global quantitative team.
I trade across prediction exchanges, domestic and international sportsbooks, focusing on platforms that attract sharp money and price markets efficiently. I participate in quantitative wagering communities including SBR Forum and r/AlgoBetting, and study line movement and market dynamics as an ongoing part of my research process.

For questions or collaboration inquiries, reach me at Ryansean87@gmail.com
