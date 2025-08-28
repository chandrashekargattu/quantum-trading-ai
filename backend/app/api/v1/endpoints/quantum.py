"""Quantum computing endpoints for portfolio optimization."""

from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.core.security import get_current_active_user

from app.models import User
from app.quantum import QuantumPortfolioOptimizer
import numpy as np

router = APIRouter()


@router.post("/optimize-portfolio-vqe")
async def optimize_portfolio_vqe(
    returns: List[float],
    covariance: List[List[float]],
    risk_aversion: float = 0.5,
    n_qubits: int = 4,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Optimize portfolio using Variational Quantum Eigensolver (VQE).
    
    - **returns**: Expected returns for each asset
    - **covariance**: Covariance matrix of returns
    - **risk_aversion**: Risk aversion parameter (0-1)
    - **n_qubits**: Number of qubits to use
    """
    try:
        # Initialize quantum optimizer
        optimizer = QuantumPortfolioOptimizer(n_qubits=n_qubits)
        
        # Convert to numpy arrays
        returns_np = np.array(returns)
        covariance_np = np.array(covariance)
        
        # Run optimization
        result = optimizer.optimize_portfolio_vqe(
            returns=returns_np,
            covariance=covariance_np,
            risk_aversion=risk_aversion
        )
        
        return {
            "optimal_weights": result.optimal_weights.tolist(),
            "expected_return": float(result.expected_return),
            "risk": float(result.risk),
            "sharpe_ratio": float(result.sharpe_ratio),
            "quantum_advantage": float(result.quantum_advantage),
            "circuit_depth": result.circuit_depth,
            "convergence_iterations": result.convergence_iterations
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quantum optimization failed: {str(e)}"
        )


@router.post("/optimize-portfolio-qaoa")
async def optimize_portfolio_qaoa(
    returns: List[float],
    covariance: List[List[float]],
    target_return: float,
    p: int = 3,
    n_qubits: int = 4,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Optimize portfolio using Quantum Approximate Optimization Algorithm (QAOA).
    
    - **returns**: Expected returns for each asset
    - **covariance**: Covariance matrix of returns
    - **target_return**: Target portfolio return
    - **p**: QAOA circuit depth parameter
    - **n_qubits**: Number of qubits to use
    """
    try:
        optimizer = QuantumPortfolioOptimizer(n_qubits=n_qubits)
        
        returns_np = np.array(returns)
        covariance_np = np.array(covariance)
        
        result = optimizer.optimize_portfolio_qaoa(
            returns=returns_np,
            covariance=covariance_np,
            target_return=target_return,
            p=p
        )
        
        return {
            "optimal_weights": result.optimal_weights.tolist(),
            "expected_return": float(result.expected_return),
            "risk": float(result.risk),
            "sharpe_ratio": float(result.sharpe_ratio),
            "quantum_advantage": float(result.quantum_advantage),
            "circuit_depth": result.circuit_depth,
            "eigenvalue": float(result.eigenvalue)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"QAOA optimization failed: {str(e)}"
        )


@router.post("/quantum-risk-analysis")
async def quantum_risk_analysis(
    portfolio_weights: List[float],
    market_scenarios: List[List[float]],
    confidence_level: float = 0.95,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Perform quantum-enhanced risk analysis.
    
    - **portfolio_weights**: Current portfolio weights
    - **market_scenarios**: Market scenario matrix
    - **confidence_level**: Confidence level for VaR/CVaR
    """
    try:
        optimizer = QuantumPortfolioOptimizer()
        
        weights_np = np.array(portfolio_weights)
        scenarios_np = np.array(market_scenarios)
        
        risk_metrics = optimizer.quantum_risk_analysis(
            portfolio_weights=weights_np,
            market_scenarios=scenarios_np,
            confidence_level=confidence_level
        )
        
        return risk_metrics
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quantum risk analysis failed: {str(e)}"
        )
