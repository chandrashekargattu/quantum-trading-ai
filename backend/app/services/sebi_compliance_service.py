"""
SEBI Compliance and Regulatory Service

Ensures compliance with Indian market regulations:
- SEBI trading rules and limits
- Circuit breakers and price bands
- F&O position limits
- Insider trading checks
- Market manipulation detection
- Tax implications (STCG, LTCG, STT)
- Reporting requirements
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict
import json

from app.core.config import settings
from app.services.indian_market_service import IndianMarketService


class ComplianceStatus(Enum):
    """Compliance check status"""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    REVIEW_REQUIRED = "review_required"


class TaxCategory(Enum):
    """Indian tax categories for trading"""
    STCG = "short_term_capital_gains"  # < 1 year
    LTCG = "long_term_capital_gains"   # > 1 year
    BUSINESS_INCOME = "business_income"  # F&O, intraday
    SPECULATION = "speculation_income"   # Intraday equity


@dataclass
class ComplianceCheck:
    """Result of compliance check"""
    check_type: str
    status: ComplianceStatus
    message: str
    details: Dict[str, Any]
    action_required: Optional[str] = None
    regulatory_reference: Optional[str] = None


@dataclass
class TaxCalculation:
    """Tax calculation for trades"""
    trade_type: str
    tax_category: TaxCategory
    gross_profit: float
    tax_rate: float
    tax_amount: float
    stt_paid: float
    net_profit: float
    details: Dict[str, Any]


class SEBIComplianceService:
    """SEBI compliance and regulatory service"""
    
    def __init__(self):
        self.market_service = IndianMarketService()
        
        # SEBI limits
        self.position_limits = {
            # Index F&O limits (lots)
            'NIFTY': {
                'client_futures': 20000,
                'client_options': 50000,
                'market_wide': 500000
            },
            'BANKNIFTY': {
                'client_futures': 5000,
                'client_options': 20000,
                'market_wide': 200000
            },
            # Stock F&O as percentage of free float
            'stock_futures': 0.20,  # 20% of free float
            'stock_options': 0.50   # 50% of free float
        }
        
        # Circuit limits
        self.circuit_filters = {
            'default': 0.20,      # 20% daily limit
            'index': 0.10,        # 10% for index
            't2t': 0.05,          # 5% for trade-to-trade
            'newly_listed': 0.20  # 20% for IPOs
        }
        
        # Tax rates (2024)
        self.tax_rates = {
            'stcg_equity': 0.15,      # 15% on equity
            'stcg_other': 0.30,       # 30% on others (slab rate)
            'ltcg_equity': 0.10,      # 10% on equity > 1 lakh
            'ltcg_exemption': 100000, # 1 lakh exemption
            'business_income': 0.30,   # 30% slab rate (simplified)
            'stt_equity_delivery': 0.001,    # 0.1% on both sides
            'stt_equity_intraday': 0.00025, # 0.025% on sell side
            'stt_options': 0.0005,    # 0.05% on premium
            'stt_futures': 0.00125    # 0.0125% on sell side
        }
        
        # Insider trading keywords
        self.insider_keywords = [
            'unpublished', 'price sensitive', 'insider information',
            'board meeting', 'acquisition', 'merger', 'results',
            'material information', 'confidential'
        ]
        
        # Market manipulation patterns
        self.manipulation_patterns = {
            'pump_and_dump': {
                'volume_spike': 5.0,  # 5x normal volume
                'price_spike': 0.10,  # 10% sudden move
                'description': 'Artificial price inflation followed by selling'
            },
            'circular_trading': {
                'self_trade_ratio': 0.20,  # 20% self trades
                'description': 'Trading among connected parties'
            },
            'spoofing': {
                'order_cancel_ratio': 0.90,  # 90% orders cancelled
                'description': 'Placing fake orders to mislead'
            },
            'front_running': {
                'timing_threshold': 60,  # Orders within 60 seconds
                'description': 'Trading ahead of large orders'
            }
        }
        
        # Reporting thresholds
        self.reporting_requirements = {
            'cash_transaction': 1000000,     # 10 lakh cash
            'suspicious_transaction': 500000, # 5 lakh unusual
            'ctr_threshold': 1000000,        # Cash Transaction Report
            'str_threshold': 500000          # Suspicious Transaction Report
        }
    
    async def check_trade_compliance(
        self, trade: Dict[str, Any], portfolio: Dict[str, Any]
    ) -> List[ComplianceCheck]:
        """Comprehensive compliance check for a trade"""
        
        checks = []
        
        # Position limit check
        position_check = await self._check_position_limits(trade, portfolio)
        checks.append(position_check)
        
        # Circuit breaker check
        circuit_check = await self._check_circuit_limits(trade)
        checks.append(circuit_check)
        
        # Margin requirement check
        margin_check = await self._check_margin_requirements(trade, portfolio)
        checks.append(margin_check)
        
        # Insider trading check
        insider_check = await self._check_insider_trading(trade)
        checks.append(insider_check)
        
        # Market manipulation check
        manipulation_check = await self._check_market_manipulation(trade, portfolio)
        checks.append(manipulation_check)
        
        # Tax implications
        tax_check = await self._check_tax_implications(trade)
        checks.append(tax_check)
        
        # Reporting requirements
        reporting_check = await self._check_reporting_requirements(trade)
        checks.append(reporting_check)
        
        return checks
    
    async def _check_position_limits(
        self, trade: Dict[str, Any], portfolio: Dict[str, Any]
    ) -> ComplianceCheck:
        """Check SEBI position limits"""
        
        symbol = trade['symbol']
        quantity = trade['quantity']
        
        # Get current positions
        current_position = portfolio.get('positions', {}).get(symbol, {})
        current_quantity = current_position.get('quantity', 0)
        
        # Calculate new position
        if trade['action'] == 'buy':
            new_position = current_quantity + quantity
        else:
            new_position = current_quantity - quantity
        
        # Check limits based on instrument type
        if symbol in ['NIFTY', 'BANKNIFTY']:
            # Index limits
            limits = self.position_limits[symbol]
            
            if trade['instrument_type'] == 'futures':
                limit = limits['client_futures']
            else:
                limit = limits['client_options']
            
            if abs(new_position) > limit:
                return ComplianceCheck(
                    check_type='position_limit',
                    status=ComplianceStatus.VIOLATION,
                    message=f"Position limit exceeded for {symbol}",
                    details={
                        'current_position': current_quantity,
                        'trade_quantity': quantity,
                        'new_position': new_position,
                        'limit': limit
                    },
                    action_required="Reduce position size",
                    regulatory_reference="SEBI Circular SEBI/HO/MRD/DP/CIR/P/2021/62"
                )
            elif abs(new_position) > limit * 0.8:
                return ComplianceCheck(
                    check_type='position_limit',
                    status=ComplianceStatus.WARNING,
                    message=f"Approaching position limit for {symbol}",
                    details={
                        'position_percentage': abs(new_position) / limit * 100
                    }
                )
        
        else:
            # Stock F&O limits based on market-wide
            # This would need actual free float data
            estimated_limit = 1000000  # Simplified
            
            if abs(new_position) > estimated_limit:
                return ComplianceCheck(
                    check_type='position_limit',
                    status=ComplianceStatus.WARNING,
                    message="Check market-wide position limits",
                    details={'position': new_position}
                )
        
        return ComplianceCheck(
            check_type='position_limit',
            status=ComplianceStatus.COMPLIANT,
            message="Within position limits",
            details={'position': new_position}
        )
    
    async def _check_circuit_limits(self, trade: Dict[str, Any]) -> ComplianceCheck:
        """Check if trade violates circuit limits"""
        
        symbol = trade['symbol']
        price = trade['price']
        
        # Get stock data
        stock_data = await self.market_service.get_stock_data(symbol)
        
        if stock_data:
            # Check if hitting circuit
            if stock_data.upper_circuit and price >= stock_data.upper_circuit:
                return ComplianceCheck(
                    check_type='circuit_limit',
                    status=ComplianceStatus.VIOLATION,
                    message=f"{symbol} at upper circuit",
                    details={
                        'price': price,
                        'upper_circuit': stock_data.upper_circuit,
                        'lower_circuit': stock_data.lower_circuit
                    },
                    action_required="Cannot place buy orders at upper circuit",
                    regulatory_reference="SEBI Price Band Framework"
                )
            
            elif stock_data.lower_circuit and price <= stock_data.lower_circuit:
                return ComplianceCheck(
                    check_type='circuit_limit',
                    status=ComplianceStatus.VIOLATION,
                    message=f"{symbol} at lower circuit",
                    details={
                        'price': price,
                        'upper_circuit': stock_data.upper_circuit,
                        'lower_circuit': stock_data.lower_circuit
                    },
                    action_required="Cannot place sell orders at lower circuit"
                )
            
            # Warning if close to circuit
            circuit_range = stock_data.upper_circuit - stock_data.lower_circuit
            distance_to_upper = (stock_data.upper_circuit - price) / circuit_range
            distance_to_lower = (price - stock_data.lower_circuit) / circuit_range
            
            if distance_to_upper < 0.1 or distance_to_lower < 0.1:
                return ComplianceCheck(
                    check_type='circuit_limit',
                    status=ComplianceStatus.WARNING,
                    message="Close to circuit limit",
                    details={
                        'distance_to_upper': f"{distance_to_upper*100:.1f}%",
                        'distance_to_lower': f"{distance_to_lower*100:.1f}%"
                    }
                )
        
        return ComplianceCheck(
            check_type='circuit_limit',
            status=ComplianceStatus.COMPLIANT,
            message="Within circuit limits",
            details={}
        )
    
    async def _check_margin_requirements(
        self, trade: Dict[str, Any], portfolio: Dict[str, Any]
    ) -> ComplianceCheck:
        """Check margin requirements"""
        
        # Calculate required margin
        if trade['instrument_type'] == 'equity':
            if trade['product_type'] == 'delivery':
                margin_required = trade['price'] * trade['quantity']
            else:  # Intraday
                margin_required = trade['price'] * trade['quantity'] * 0.20  # 20% margin
        
        elif trade['instrument_type'] in ['futures', 'options']:
            # F&O margin (simplified)
            margin_required = trade['price'] * trade['quantity'] * 0.15
        
        else:
            margin_required = trade['price'] * trade['quantity']
        
        # Check available margin
        available_margin = portfolio.get('available_margin', 0)
        
        if margin_required > available_margin:
            return ComplianceCheck(
                check_type='margin_requirement',
                status=ComplianceStatus.VIOLATION,
                message="Insufficient margin",
                details={
                    'required_margin': margin_required,
                    'available_margin': available_margin,
                    'shortfall': margin_required - available_margin
                },
                action_required="Add funds or reduce position size",
                regulatory_reference="SEBI Margin Trading Regulations"
            )
        
        elif margin_required > available_margin * 0.8:
            return ComplianceCheck(
                check_type='margin_requirement',
                status=ComplianceStatus.WARNING,
                message="High margin utilization",
                details={
                    'margin_utilization': f"{margin_required/available_margin*100:.1f}%"
                }
            )
        
        return ComplianceCheck(
            check_type='margin_requirement',
            status=ComplianceStatus.COMPLIANT,
            message="Adequate margin available",
            details={
                'margin_utilization': f"{margin_required/available_margin*100:.1f}%"
            }
        )
    
    async def _check_insider_trading(self, trade: Dict[str, Any]) -> ComplianceCheck:
        """Check for potential insider trading patterns"""
        
        # Check if user has insider designation
        if trade.get('user_designation') in ['director', 'key_managerial', 'promoter']:
            # Check trading window
            if self._is_trading_window_closed(trade['symbol']):
                return ComplianceCheck(
                    check_type='insider_trading',
                    status=ComplianceStatus.VIOLATION,
                    message="Trading window closed for insiders",
                    details={
                        'designation': trade['user_designation'],
                        'symbol': trade['symbol']
                    },
                    action_required="Wait for trading window to open",
                    regulatory_reference="SEBI (PIT) Regulations, 2015"
                )
        
        # Check for suspicious patterns
        if trade.get('order_source') == 'manual':
            # Check trade timing relative to news
            if await self._check_pre_news_trading(trade):
                return ComplianceCheck(
                    check_type='insider_trading',
                    status=ComplianceStatus.REVIEW_REQUIRED,
                    message="Trade placed before material news",
                    details={
                        'time_before_news': '30 minutes',
                        'news_type': 'earnings'
                    },
                    action_required="Review for insider trading compliance"
                )
        
        return ComplianceCheck(
            check_type='insider_trading',
            status=ComplianceStatus.COMPLIANT,
            message="No insider trading concerns",
            details={}
        )
    
    async def _check_market_manipulation(
        self, trade: Dict[str, Any], portfolio: Dict[str, Any]
    ) -> ComplianceCheck:
        """Check for market manipulation patterns"""
        
        # Get recent trades
        recent_trades = portfolio.get('recent_trades', [])
        symbol_trades = [t for t in recent_trades if t['symbol'] == trade['symbol']]
        
        # Check for pump and dump
        if len(symbol_trades) > 10:
            # High frequency trading in same symbol
            avg_price = np.mean([t['price'] for t in symbol_trades])
            price_change = (trade['price'] - avg_price) / avg_price
            
            if price_change > 0.10 and trade['action'] == 'sell':
                return ComplianceCheck(
                    check_type='market_manipulation',
                    status=ComplianceStatus.WARNING,
                    message="Potential pump and dump pattern",
                    details={
                        'price_increase': f"{price_change*100:.1f}%",
                        'trades_count': len(symbol_trades)
                    },
                    action_required="Review trading pattern"
                )
        
        # Check for circular trading
        if self._check_circular_trading(recent_trades):
            return ComplianceCheck(
                check_type='market_manipulation',
                status=ComplianceStatus.WARNING,
                message="Potential circular trading detected",
                details={
                    'pattern': 'Repeated buy/sell with minimal price change'
                }
            )
        
        # Check order cancellation ratio
        cancelled_orders = portfolio.get('cancelled_orders', [])
        total_orders = len(recent_trades) + len(cancelled_orders)
        
        if total_orders > 0:
            cancel_ratio = len(cancelled_orders) / total_orders
            
            if cancel_ratio > 0.90:
                return ComplianceCheck(
                    check_type='market_manipulation',
                    status=ComplianceStatus.WARNING,
                    message="High order cancellation rate",
                    details={
                        'cancellation_rate': f"{cancel_ratio*100:.1f}%"
                    },
                    regulatory_reference="SEBI Order Spoofing Guidelines"
                )
        
        return ComplianceCheck(
            check_type='market_manipulation',
            status=ComplianceStatus.COMPLIANT,
            message="No manipulation patterns detected",
            details={}
        )
    
    async def _check_tax_implications(self, trade: Dict[str, Any]) -> ComplianceCheck:
        """Check tax implications of trade"""
        
        # Determine tax category
        if trade['instrument_type'] in ['futures', 'options']:
            tax_category = TaxCategory.BUSINESS_INCOME
            tax_rate = self.tax_rates['business_income']
        
        elif trade['product_type'] == 'intraday':
            tax_category = TaxCategory.SPECULATION
            tax_rate = self.tax_rates['business_income']
        
        else:
            # Equity delivery
            holding_period = trade.get('holding_period_days', 0)
            
            if holding_period > 365:
                tax_category = TaxCategory.LTCG
                tax_rate = self.tax_rates['ltcg_equity']
            else:
                tax_category = TaxCategory.STCG
                tax_rate = self.tax_rates['stcg_equity']
        
        # Calculate STT
        stt = self._calculate_stt(trade)
        
        # Estimate tax impact
        if trade['action'] == 'sell':
            profit = (trade['price'] - trade.get('buy_price', trade['price'])) * trade['quantity']
            tax_amount = profit * tax_rate if profit > 0 else 0
            
            details = {
                'tax_category': tax_category.value,
                'tax_rate': f"{tax_rate*100:.1f}%",
                'estimated_tax': tax_amount,
                'stt': stt,
                'net_profit': profit - tax_amount - stt
            }
            
            # Add LTCG exemption if applicable
            if tax_category == TaxCategory.LTCG and profit > self.tax_rates['ltcg_exemption']:
                details['ltcg_exemption'] = self.tax_rates['ltcg_exemption']
                details['taxable_ltcg'] = profit - self.tax_rates['ltcg_exemption']
        
        else:
            details = {
                'tax_category': 'To be determined on sale',
                'stt_paid': stt
            }
        
        return ComplianceCheck(
            check_type='tax_implications',
            status=ComplianceStatus.COMPLIANT,
            message=f"Tax category: {tax_category.value if trade['action'] == 'sell' else 'TBD'}",
            details=details
        )
    
    async def _check_reporting_requirements(
        self, trade: Dict[str, Any]
    ) -> ComplianceCheck:
        """Check reporting requirements"""
        
        trade_value = trade['price'] * trade['quantity']
        
        # Check for large cash transactions
        if trade.get('payment_mode') == 'cash' and trade_value > self.reporting_requirements['cash_transaction']:
            return ComplianceCheck(
                check_type='reporting_requirement',
                status=ComplianceStatus.WARNING,
                message="Cash transaction exceeds reporting threshold",
                details={
                    'transaction_value': trade_value,
                    'threshold': self.reporting_requirements['cash_transaction'],
                    'report_type': 'CTR'
                },
                action_required="File Cash Transaction Report",
                regulatory_reference="PMLA Requirements"
            )
        
        # Check for suspicious transactions
        if self._is_suspicious_transaction(trade):
            return ComplianceCheck(
                check_type='reporting_requirement',
                status=ComplianceStatus.REVIEW_REQUIRED,
                message="Transaction requires review",
                details={
                    'reason': 'Unusual pattern detected',
                    'report_type': 'STR'
                },
                action_required="Review and file STR if required"
            )
        
        return ComplianceCheck(
            check_type='reporting_requirement',
            status=ComplianceStatus.COMPLIANT,
            message="No reporting required",
            details={}
        )
    
    def _is_trading_window_closed(self, symbol: str) -> bool:
        """Check if trading window is closed for insiders"""
        
        # Check for quarterly results period
        current_date = datetime.now()
        quarter_end_months = [3, 6, 9, 12]
        
        if current_date.month in quarter_end_months:
            # Trading window typically closed from quarter end till results
            if current_date.day >= 25 or (current_date.month % 3 == 1 and current_date.day <= 15):
                return True
        
        return False
    
    async def _check_pre_news_trading(self, trade: Dict[str, Any]) -> bool:
        """Check if trade was placed before material news"""
        
        # This would check actual news timestamps
        # For now, simplified check
        return False
    
    def _check_circular_trading(self, trades: List[Dict]) -> bool:
        """Check for circular trading patterns"""
        
        if len(trades) < 4:
            return False
        
        # Look for buy-sell patterns with minimal price change
        for i in range(len(trades) - 3):
            if (trades[i]['action'] == 'buy' and 
                trades[i+1]['action'] == 'sell' and
                trades[i+2]['action'] == 'buy' and
                trades[i+3]['action'] == 'sell'):
                
                # Check price change
                prices = [t['price'] for t in trades[i:i+4]]
                price_change = (max(prices) - min(prices)) / min(prices)
                
                if price_change < 0.02:  # Less than 2% change
                    return True
        
        return False
    
    def _calculate_stt(self, trade: Dict[str, Any]) -> float:
        """Calculate Securities Transaction Tax"""
        
        trade_value = trade['price'] * trade['quantity']
        
        if trade['instrument_type'] == 'equity':
            if trade['product_type'] == 'delivery':
                # Both buy and sell
                stt = trade_value * self.tax_rates['stt_equity_delivery']
            else:
                # Intraday - only on sell
                if trade['action'] == 'sell':
                    stt = trade_value * self.tax_rates['stt_equity_intraday']
                else:
                    stt = 0
        
        elif trade['instrument_type'] == 'options':
            # On premium
            stt = trade_value * self.tax_rates['stt_options']
        
        elif trade['instrument_type'] == 'futures':
            # On sell side
            if trade['action'] == 'sell':
                stt = trade_value * self.tax_rates['stt_futures']
            else:
                stt = 0
        
        else:
            stt = 0
        
        return stt
    
    def _is_suspicious_transaction(self, trade: Dict[str, Any]) -> bool:
        """Check if transaction is suspicious"""
        
        # Unusual size
        if trade['quantity'] > 10000:
            return True
        
        # Unusual timing
        trade_time = datetime.fromisoformat(trade.get('timestamp', datetime.now().isoformat()))
        if trade_time.hour < 9 or trade_time.hour > 15:
            return True
        
        # Price far from market
        if trade.get('price_deviation', 0) > 0.05:  # 5% from market
            return True
        
        return False
    
    async def calculate_tax_liability(
        self, trades: List[Dict[str, Any]], financial_year: str = "2023-24"
    ) -> Dict[str, Any]:
        """Calculate total tax liability for the financial year"""
        
        tax_summary = {
            'stcg': {'trades': 0, 'profit': 0, 'tax': 0},
            'ltcg': {'trades': 0, 'profit': 0, 'tax': 0},
            'business_income': {'trades': 0, 'profit': 0, 'tax': 0},
            'speculation': {'trades': 0, 'profit': 0, 'tax': 0},
            'total_stt': 0,
            'total_tax': 0,
            'net_profit': 0
        }
        
        for trade in trades:
            if trade['action'] != 'sell':
                continue
            
            # Calculate profit
            profit = (trade['sell_price'] - trade['buy_price']) * trade['quantity']
            stt = self._calculate_stt(trade)
            
            # Categorize trade
            if trade['instrument_type'] in ['futures', 'options']:
                category = 'business_income'
                tax_rate = self.tax_rates['business_income']
            
            elif trade['product_type'] == 'intraday':
                category = 'speculation'
                tax_rate = self.tax_rates['business_income']
            
            else:
                holding_days = (datetime.fromisoformat(trade['sell_date']) - 
                              datetime.fromisoformat(trade['buy_date'])).days
                
                if holding_days > 365:
                    category = 'ltcg'
                    tax_rate = self.tax_rates['ltcg_equity']
                else:
                    category = 'stcg'
                    tax_rate = self.tax_rates['stcg_equity']
            
            # Update summary
            tax_summary[category]['trades'] += 1
            tax_summary[category]['profit'] += profit
            tax_summary['total_stt'] += stt
            
            if profit > 0:
                # Apply exemptions for LTCG
                if category == 'ltcg':
                    taxable_profit = max(0, tax_summary['ltcg']['profit'] - self.tax_rates['ltcg_exemption'])
                    tax = taxable_profit * tax_rate
                else:
                    tax = profit * tax_rate
                
                tax_summary[category]['tax'] += tax
                tax_summary['total_tax'] += tax
        
        # Calculate net profit
        total_profit = sum(cat['profit'] for cat in 
                          [tax_summary['stcg'], tax_summary['ltcg'], 
                           tax_summary['business_income'], tax_summary['speculation']])
        
        tax_summary['net_profit'] = total_profit - tax_summary['total_tax'] - tax_summary['total_stt']
        
        # Add recommendations
        tax_summary['recommendations'] = self._get_tax_recommendations(tax_summary)
        
        return tax_summary
    
    def _get_tax_recommendations(self, tax_summary: Dict[str, Any]) -> List[str]:
        """Get tax optimization recommendations"""
        
        recommendations = []
        
        # LTCG optimization
        if tax_summary['stcg']['profit'] > 100000:
            recommendations.append(
                "Consider holding positions for >1 year to qualify for lower LTCG tax rate"
            )
        
        # Business income vs capital gains
        if tax_summary['business_income']['trades'] > 50:
            recommendations.append(
                "High F&O trading frequency - ensure proper books of accounts as per Income Tax Act"
            )
        
        # Tax harvesting
        if tax_summary['ltcg']['profit'] > self.tax_rates['ltcg_exemption']:
            recommendations.append(
                "Consider tax-loss harvesting to offset LTCG above ₹1 lakh exemption"
            )
        
        # Speculation income
        if tax_summary['speculation']['profit'] < 0:
            recommendations.append(
                "Speculation losses can only be set off against speculation income"
            )
        
        return recommendations
    
    async def generate_compliance_report(
        self, portfolio: Dict[str, Any], period: str = "quarterly"
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        
        report = {
            'report_date': datetime.now().isoformat(),
            'period': period,
            'portfolio_id': portfolio['id'],
            'compliance_score': 0,
            'violations': [],
            'warnings': [],
            'tax_summary': {},
            'reporting_obligations': [],
            'recommendations': []
        }
        
        # Check all positions for compliance
        all_checks = []
        
        for position in portfolio.get('positions', []):
            checks = await self.check_trade_compliance(
                {'symbol': position['symbol'], 
                 'quantity': position['quantity'],
                 'price': position['current_price'],
                 'action': 'hold',
                 'instrument_type': position.get('instrument_type', 'equity')},
                portfolio
            )
            all_checks.extend(checks)
        
        # Categorize issues
        for check in all_checks:
            if check.status == ComplianceStatus.VIOLATION:
                report['violations'].append({
                    'type': check.check_type,
                    'message': check.message,
                    'action_required': check.action_required
                })
            elif check.status == ComplianceStatus.WARNING:
                report['warnings'].append({
                    'type': check.check_type,
                    'message': check.message
                })
        
        # Calculate compliance score
        total_checks = len(all_checks)
        compliant_checks = sum(1 for c in all_checks if c.status == ComplianceStatus.COMPLIANT)
        
        if total_checks > 0:
            report['compliance_score'] = (compliant_checks / total_checks) * 100
        
        # Add tax summary
        trades = portfolio.get('trades', [])
        report['tax_summary'] = await self.calculate_tax_liability(trades)
        
        # Add recommendations
        if report['compliance_score'] < 80:
            report['recommendations'].append(
                "Review and address compliance violations immediately"
            )
        
        if report['warnings']:
            report['recommendations'].append(
                "Monitor warning areas to prevent future violations"
            )
        
        return report
