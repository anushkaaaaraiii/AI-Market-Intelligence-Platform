import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { AlertTriangle, TrendingDown, Activity, Shield, RefreshCw, Briefcase } from 'lucide-react';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { getHistoricalData } from '../services/api';

const PORTFOLIO_KEY = 'sentinelquant:portfolio';
const SIMS = 5000;
const MONTHS = 6;
const TRADING_DAYS_PER_MONTH = 21;

interface Holding {
  symbol: string;
  name: string;
  sector?: string;
  quantity: number;
  avgPrice: number;
  currentPrice?: number;
}

interface SimResult {
  monteCarloData: { month: number; p5: number; p25: number; p50: number; p75: number; p95: number }[];
  var95: number;
  var99: number;
  cvar95: number;
  maxDrawdown: number;
  beta: number;
  semiDeviation: number;
  drawdownProbs: { threshold: string; probability: number }[];
  stressTests: { scenario: string; portfolioImpact: number; varImpact: number; recovery: string }[];
  insights: { icon: 'warn' | 'info' | 'good'; title: string; text: string }[];
  totalValue: number;
}

function percentile(sorted: number[], p: number): number {
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function runMonteCarlo(
  weights: number[],
  returns: number[][],
  totalValue: number,
): SimResult {
  // Compute weighted portfolio returns
  const minLen = Math.min(...returns.map(r => r.length));
  const portfolioReturns: number[] = [];
  for (let d = 0; d < minLen; d++) {
    let r = 0;
    for (let s = 0; s < weights.length; s++) {
      r += weights[s] * (returns[s][d] ?? 0);
    }
    portfolioReturns.push(r);
  }

  const meanReturn = portfolioReturns.reduce((a, b) => a + b, 0) / portfolioReturns.length;
  const stdDev = Math.sqrt(
    portfolioReturns.reduce((a, r) => a + (r - meanReturn) ** 2, 0) / (portfolioReturns.length - 1)
  );

  // Semi-deviation (downside only)
  const downsideReturns = portfolioReturns.filter(r => r < 0);
  const semiDeviation = downsideReturns.length > 0
    ? Math.sqrt(downsideReturns.reduce((a, r) => a + r ** 2, 0) / downsideReturns.length) * Math.sqrt(252) * 100
    : 0;

  // Beta (approximate vs equal-weight market)
  const marketReturns: number[] = [];
  for (let d = 0; d < minLen; d++) {
    let r = 0;
    for (let s = 0; s < returns.length; s++) {
      r += (returns[s][d] ?? 0) / returns.length;
    }
    marketReturns.push(r);
  }
  const marketMean = marketReturns.reduce((a, b) => a + b, 0) / marketReturns.length;
  let cov = 0, varM = 0;
  for (let d = 0; d < minLen; d++) {
    cov += (portfolioReturns[d] - meanReturn) * (marketReturns[d] - marketMean);
    varM += (marketReturns[d] - marketMean) ** 2;
  }
  const beta = varM > 0 ? cov / varM : 1;

  // Max historical drawdown
  let peak = 1, maxDD = 0, cum = 1;
  for (const r of portfolioReturns) {
    cum *= (1 + r);
    if (cum > peak) peak = cum;
    const dd = (peak - cum) / peak;
    if (dd > maxDD) maxDD = dd;
  }

  // Monte Carlo simulation
  const dailyMu = meanReturn;
  const dailySigma = stdDev;
  const totalDays = MONTHS * TRADING_DAYS_PER_MONTH;
  const finalValues: number[] = [];
  const pathsByMonth: number[][] = Array.from({ length: MONTHS + 1 }, () => []);

  for (let sim = 0; sim < SIMS; sim++) {
    let value = totalValue;
    pathsByMonth[0].push(value);
    for (let m = 1; m <= MONTHS; m++) {
      for (let d = 0; d < TRADING_DAYS_PER_MONTH; d++) {
        // Box-Muller transform for normal random
        const u1 = Math.random();
        const u2 = Math.random();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        const dailyReturn = dailyMu + dailySigma * z;
        value *= (1 + dailyReturn);
      }
      pathsByMonth[m].push(value);
    }
    finalValues.push(value);
  }

  // Build percentile paths
  const monteCarloData = pathsByMonth.map((values, month) => {
    const sorted = [...values].sort((a, b) => a - b);
    return {
      month,
      p5: percentile(sorted, 5),
      p25: percentile(sorted, 25),
      p50: percentile(sorted, 50),
      p75: percentile(sorted, 75),
      p95: percentile(sorted, 95),
    };
  });

  // VaR / CVaR from final values
  const pnl = finalValues.map(v => v - totalValue);
  const sortedPnl = [...pnl].sort((a, b) => a - b);
  const var95 = -percentile(sortedPnl, 5);
  const var99 = -percentile(sortedPnl, 1);
  const tail5 = sortedPnl.slice(0, Math.ceil(sortedPnl.length * 0.05));
  const cvar95 = tail5.length > 0 ? -(tail5.reduce((a, b) => a + b, 0) / tail5.length) : 0;

  // Drawdown probabilities
  const drawdownThresholds = [5, 10, 15, 20, 25];
  const drawdownProbs = drawdownThresholds.map(threshold => {
    const count = finalValues.filter(v => (totalValue - v) / totalValue * 100 >= threshold).length;
    return { threshold: `${threshold}%`, probability: Math.round(count / SIMS * 100) };
  });

  // Stress tests (analytical)
  const stressTests = [
    {
      scenario: 'Market Crash (-30%)',
      portfolioImpact: -(beta * 30),
      varImpact: -(beta * 30 / 100 * totalValue),
      recovery: `${Math.ceil(14 * beta)} months`,
    },
    {
      scenario: 'Sector Rotation',
      portfolioImpact: -(beta * 12),
      varImpact: -(beta * 12 / 100 * totalValue),
      recovery: `${Math.ceil(6 * beta)} months`,
    },
    {
      scenario: 'Interest Rate Hike',
      portfolioImpact: -(beta * 8),
      varImpact: -(beta * 8 / 100 * totalValue),
      recovery: `${Math.ceil(4 * beta)} months`,
    },
    {
      scenario: 'Currency Shock',
      portfolioImpact: -(beta * 5),
      varImpact: -(beta * 5 / 100 * totalValue),
      recovery: `${Math.ceil(3 * beta)} months`,
    },
  ];

  // Insights
  const insights: SimResult['insights'] = [];
  const annualizedVol = stdDev * Math.sqrt(252) * 100;

  // Concentration check from weights
  const maxWeight = Math.max(...weights) * 100;
  if (maxWeight > 30) {
    insights.push({
      icon: 'warn',
      title: 'High Concentration Risk',
      text: `Your largest position is ${maxWeight.toFixed(1)}% of the portfolio, exceeding the recommended 30% threshold. Consider rebalancing to reduce single-stock risk.`,
    });
  }

  if (beta > 1.2) {
    insights.push({
      icon: 'info',
      title: 'Above-Market Volatility',
      text: `Portfolio beta of ${beta.toFixed(2)} indicates higher volatility than the broader market. Consider adding defensive stocks to reduce beta.`,
    });
  } else if (beta < 0.8) {
    insights.push({
      icon: 'good',
      title: 'Defensive Portfolio',
      text: `Portfolio beta of ${beta.toFixed(2)} indicates lower volatility than the broader market. Good for capital preservation.`,
    });
  } else {
    insights.push({
      icon: 'info',
      title: 'Moderate Market Exposure',
      text: `Portfolio beta of ${beta.toFixed(2)} indicates market-neutral volatility. Current allocation is appropriate for moderate risk tolerance.`,
    });
  }

  if (semiDeviation < 10) {
    insights.push({
      icon: 'good',
      title: 'Good Downside Protection',
      text: `Semi-deviation of ${semiDeviation.toFixed(1)}% suggests effective downside protection compared to typical equity portfolios.`,
    });
  }

  if (annualizedVol > 25) {
    insights.push({
      icon: 'warn',
      title: 'High Portfolio Volatility',
      text: `Annualized volatility of ${annualizedVol.toFixed(1)}% is elevated. The expected monthly VaR at 95% confidence is ₹${(var95 / 100000).toFixed(2)}L.`,
    });
  }

  return {
    monteCarloData,
    var95,
    var99,
    cvar95,
    maxDrawdown: maxDD * 100,
    beta,
    semiDeviation,
    drawdownProbs,
    stressTests,
    insights,
    totalValue,
  };
}

export default function RiskSimulation() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [holdings, setHoldings] = useState<Holding[]>([]);
  const [simResult, setSimResult] = useState<SimResult | null>(null);

  const loadAndSimulate = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      // Load portfolio from localStorage
      const raw = localStorage.getItem(PORTFOLIO_KEY);
      const parsed: Holding[] = raw ? JSON.parse(raw) : [];
      setHoldings(parsed);

      if (parsed.length === 0) {
        setSimResult(null);
        setLoading(false);
        return;
      }

      // Compute total value and weights
      const values = parsed.map(h => h.quantity * (h.currentPrice ?? h.avgPrice));
      const totalValue = values.reduce((a, b) => a + b, 0);
      if (totalValue <= 0) {
        setSimResult(null);
        setLoading(false);
        return;
      }
      const weights = values.map(v => v / totalValue);

      // Fetch historical data for each holding (90 days)
      const histPromises = parsed.map(h =>
        getHistoricalData(h.symbol, 120).catch(() => null)
      );
      const histResults = await Promise.all(histPromises);

      // Compute daily returns for each stock
      const allReturns: number[][] = [];
      for (const hist of histResults) {
        if (!hist || !hist.data || hist.data.length < 10) {
          // Use zero returns as fallback
          allReturns.push(Array(90).fill(0));
          continue;
        }
        const prices = hist.data.map((d: any) => d.close);
        const returns: number[] = [];
        for (let i = 1; i < prices.length; i++) {
          if (prices[i - 1] > 0) {
            returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
          }
        }
        allReturns.push(returns);
      }

      // Run Monte Carlo
      const result = runMonteCarlo(weights, allReturns, totalValue);
      setSimResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run simulation');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAndSimulate();
  }, [loadAndSimulate]);

  const fmt = (v: number) => `₹${(v / 100000).toFixed(2)}L`;
  const fmtK = (v: number) => `₹${(Math.abs(v) / 1000).toFixed(0)}K`;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-6 h-6 animate-spin text-primary" />
        <span className="ml-3 text-muted-foreground">Running {SIMS.toLocaleString()} Monte Carlo simulations…</span>
      </div>
    );
  }

  if (!simResult || holdings.length === 0) {
    return (
      <div className="space-y-6 max-w-[1600px] mx-auto">
        <div>
          <h1 className="text-3xl mb-2">Risk & Simulation</h1>
          <p className="text-muted-foreground">Monte Carlo simulations and risk analysis</p>
        </div>
        <div className="glass-card rounded-xl p-12 border border-white/10 text-center">
          <Briefcase className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-xl mb-2">No Portfolio Found</h3>
          <p className="text-muted-foreground mb-4">
            Add stocks to your portfolio first to run risk simulations.
          </p>
          <Button onClick={() => window.location.href = '/portfolio'} variant="outline">
            Go to Portfolio
          </Button>
        </div>
      </div>
    );
  }

  const { monteCarloData, var95, var99, cvar95, maxDrawdown, beta, semiDeviation, drawdownProbs, stressTests, insights, totalValue } = simResult;

  return (
    <div className="space-y-6 max-w-[1600px] mx-auto">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl mb-2">Risk & Simulation</h1>
          <p className="text-muted-foreground">
            {SIMS.toLocaleString()} Monte Carlo simulations · {holdings.length} holdings · {MONTHS}-month projection
          </p>
        </div>
        <Button variant="outline" onClick={loadAndSimulate} className="border-white/10">
          <RefreshCw className="w-4 h-4 mr-2" /> Re-run
        </Button>
      </div>

      {error && (
        <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/30 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Risk Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-card rounded-xl p-6 border border-white/10">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-muted-foreground">Portfolio VaR (95%)</span>
            <AlertTriangle className="w-5 h-5 text-destructive" />
          </div>
          <div className="text-2xl mb-1 text-destructive">{fmt(var95)}</div>
          <div className="text-xs text-muted-foreground">{MONTHS}-month horizon</div>
        </div>

        <div className="glass-card rounded-xl p-6 border border-white/10">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-muted-foreground">Max Drawdown</span>
            <TrendingDown className="w-5 h-5 text-warning" />
          </div>
          <div className="text-2xl mb-1 text-warning">-{maxDrawdown.toFixed(1)}%</div>
          <div className="text-xs text-muted-foreground">Historical peak to trough</div>
        </div>

        <div className="glass-card rounded-xl p-6 border border-white/10">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-muted-foreground">Beta</span>
            <Activity className="w-5 h-5 text-primary" />
          </div>
          <div className="text-2xl mb-1 text-primary">{beta.toFixed(2)}</div>
          <div className="text-xs text-muted-foreground">vs. portfolio equal-weight</div>
        </div>

        <div className="glass-card rounded-xl p-6 border border-white/10">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-muted-foreground">Downside Risk</span>
            <Shield className="w-5 h-5 text-success" />
          </div>
          <div className="text-2xl mb-1 text-success">{semiDeviation.toFixed(1)}%</div>
          <div className="text-xs text-muted-foreground">Semi-deviation (annualized)</div>
        </div>
      </div>

      {/* Monte Carlo Simulation */}
      <div className="glass-card rounded-xl p-6 border border-white/10">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="mb-1">Monte Carlo Simulation</h3>
            <p className="text-sm text-muted-foreground">
              {SIMS.toLocaleString()} simulations · Portfolio value: {fmt(totalValue)}
            </p>
          </div>
          <Badge className="bg-accent/20 text-accent border-accent/30">Live</Badge>
        </div>
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={monteCarloData}>
            <defs>
              <linearGradient id="confidence95" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.1}/>
                <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.05}/>
              </linearGradient>
              <linearGradient id="confidence75" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.2}/>
                <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.1}/>
              </linearGradient>
              <linearGradient id="confidence50" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.15}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.1)" />
            <XAxis
              dataKey="month"
              stroke="#94a3b8"
              fontSize={12}
              label={{ value: 'Months', position: 'insideBottom', offset: -5, fill: '#94a3b8' }}
            />
            <YAxis
              stroke="#94a3b8"
              fontSize={12}
              tickFormatter={(value: number) => `₹${(value / 100000).toFixed(1)}L`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(15, 23, 42, 0.9)',
                border: '1px solid rgba(148, 163, 184, 0.1)',
                borderRadius: '8px'
              }}
              formatter={(value: number) => `₹${(value / 100000).toFixed(2)}L`}
            />
            <Area type="monotone" dataKey="p95" stroke="transparent" fill="url(#confidence95)" fillOpacity={1} />
            <Area type="monotone" dataKey="p75" stroke="transparent" fill="url(#confidence75)" fillOpacity={1} />
            <Area type="monotone" dataKey="p50" stroke="#06b6d4" fill="url(#confidence50)" fillOpacity={1} strokeWidth={2} />
            <Area type="monotone" dataKey="p25" stroke="transparent" fill="url(#confidence75)" fillOpacity={1} />
            <Area type="monotone" dataKey="p5" stroke="transparent" fill="url(#confidence95)" fillOpacity={1} />
            <ReferenceLine y={totalValue} stroke="#8b5cf6" strokeDasharray="3 3" label="Current Value" />
          </AreaChart>
        </ResponsiveContainer>
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mt-6">
          {[
            { label: '5th Percentile', key: 'p5' as const },
            { label: '25th Percentile', key: 'p25' as const },
            { label: 'Median', key: 'p50' as const, highlight: true },
            { label: '75th Percentile', key: 'p75' as const },
            { label: '95th Percentile', key: 'p95' as const },
          ].map(({ label, key, highlight }) => {
            const last = monteCarloData[monteCarloData.length - 1];
            return (
              <div key={key} className={`p-3 rounded-lg ${highlight ? 'bg-primary/20 border border-primary/30' : 'bg-muted/30'}`}>
                <div className="text-xs text-muted-foreground mb-1">{label}</div>
                <div className={`text-sm ${highlight ? 'text-primary' : ''}`}>{fmt(last[key])}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Drawdown Probability + VaR */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-card rounded-xl p-6 border border-white/10">
          <h3 className="mb-4">Drawdown Probability</h3>
          <div className="space-y-4">
            {drawdownProbs.map(item => (
              <div key={item.threshold}>
                <div className="flex justify-between text-sm mb-2">
                  <span>{item.threshold} Drawdown</span>
                  <span className="text-muted-foreground">{item.probability}% probability</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-warning to-destructive rounded-full transition-all"
                    style={{ width: `${Math.min(item.probability * 3, 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="glass-card rounded-xl p-6 border border-white/10">
          <h3 className="mb-4">Value at Risk (VaR)</h3>
          <div className="space-y-4">
            <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/30">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm">95% Confidence ({MONTHS}-month)</span>
                <Badge className="bg-destructive/20 text-destructive border-destructive/30">High Risk</Badge>
              </div>
              <div className="text-2xl text-destructive">{fmt(var95)}</div>
              <p className="text-xs text-muted-foreground mt-2">
                95% probability that losses won't exceed this amount
              </p>
            </div>

            <div className="p-4 rounded-lg bg-warning/10 border border-warning/30">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm">99% Confidence ({MONTHS}-month)</span>
                <Badge className="bg-warning/20 text-warning border-warning/30">Very High Risk</Badge>
              </div>
              <div className="text-2xl text-warning">{fmt(var99)}</div>
              <p className="text-xs text-muted-foreground mt-2">
                99% probability that losses won't exceed this amount
              </p>
            </div>

            <div className="p-4 rounded-lg bg-primary/10 border border-primary/30">
              <div className="text-sm text-muted-foreground mb-1">Expected Shortfall (CVaR)</div>
              <div className="text-xl text-primary">{fmt(cvar95)}</div>
              <p className="text-xs text-muted-foreground mt-2">
                Average loss beyond VaR threshold
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Stress Test Scenarios */}
      <div className="glass-card rounded-xl border border-white/10 overflow-hidden">
        <div className="p-6 border-b border-white/10">
          <h3>Stress Test Scenarios</h3>
          <p className="text-sm text-muted-foreground mt-1">Portfolio resilience under adverse market conditions (β-adjusted)</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-white/10 bg-muted/30">
                <th className="text-left p-4 text-sm">Scenario</th>
                <th className="text-right p-4 text-sm">Portfolio Impact</th>
                <th className="text-right p-4 text-sm">Estimated Loss</th>
                <th className="text-right p-4 text-sm">Expected Recovery</th>
              </tr>
            </thead>
            <tbody>
              {stressTests.map(test => (
                <tr key={test.scenario} className="border-b border-white/5 hover:bg-white/5 transition-smooth">
                  <td className="p-4">{test.scenario}</td>
                  <td className="p-4 text-right">
                    <span className="text-destructive">{test.portfolioImpact.toFixed(1)}%</span>
                  </td>
                  <td className="p-4 text-right text-destructive">
                    {Math.abs(test.varImpact) > 100000 ? fmt(Math.abs(test.varImpact)) : fmtK(test.varImpact)}
                  </td>
                  <td className="p-4 text-right text-muted-foreground">{test.recovery}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* AI Risk Insights */}
      {insights.length > 0 && (
        <div className="glass-card rounded-xl p-6 border border-accent/30 glow-accent">
          <div className="flex items-center gap-2 mb-4">
            <Shield className="w-5 h-5 text-accent" />
            <h3>AI Risk Insights</h3>
          </div>
          <div className="space-y-3">
            {insights.map((insight, i) => (
              <div key={i} className="p-4 rounded-lg bg-muted/30">
                <div className="flex items-start gap-3">
                  {insight.icon === 'warn' && <AlertTriangle className="w-5 h-5 text-warning flex-shrink-0 mt-1" />}
                  {insight.icon === 'info' && <Activity className="w-5 h-5 text-primary flex-shrink-0 mt-1" />}
                  {insight.icon === 'good' && <Shield className="w-5 h-5 text-success flex-shrink-0 mt-1" />}
                  <div>
                    <div className="mb-1">{insight.title}</div>
                    <p className="text-sm text-muted-foreground">{insight.text}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
