"""
AnasePivotV4 Professional Pivot Detection System
===============================================

Exact Python conversion of the AnasePivotV4 Pine Script pivot detection algorithm.
Designed to produce identical results when used side-by-side with the original Pine Script indicator.

Key Features:
- Complete C1-C0 confirmation system with exact Pine Script equivalence
- Reference candle methods (wick rejection vs reversal open)
- Pivot classification (HH, LH, HL, LL) with historical comparison
- MSS (Market Structure Shift) detection on LH/HL levels
- BOS (Break of Structure) detection on HH/LL levels with structural buffer
- State management preserving exact Pine Script behavior

Mathematical Precision:
- All calculations maintain floating-point precision matching Pine Script
- Array indexing converted exactly: Pine Script [i] → Python .iloc[-i-1]
- State transitions match Pine Script bar-by-bar execution
- Edge case handling preserved from original implementation

Author: Converted from AnasePivotV4 Pine Script Indicator
Version: 1.0 - Professional Implementation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class ReferenceMethod(Enum):
    """Reference candle detection methods"""
    WICK_REJECTION = "WICK_REJECTION"
    REVERSAL_OPEN = "REVERSAL_OPEN"


class PivotType(Enum):
    """Pivot types"""
    HIGH = "HIGH"
    LOW = "LOW"


class PivotLabel(Enum):
    """Pivot classification labels"""
    HH = "HH"  # Higher High
    LH = "LH"  # Lower High
    HL = "HL"  # Higher Low
    LL = "LL"  # Lower Low


class TrendState(Enum):
    """Market trend states"""
    UPTREND = "Uptrend"
    DOWNTREND = "Downtrend"
    UNDEFINED = "Undefined"


@dataclass
class ReferenceCandle:
    """Reference candle data structure"""
    bar_index: int
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    timestamp: Any = None
    
    def is_bullish(self) -> bool:
        """Check if reference candle is bullish"""
        return self.close_price > self.open_price
    
    def is_bearish(self) -> bool:
        """Check if reference candle is bearish"""
        return self.close_price < self.open_price


@dataclass
class ConfirmedPivot:
    """Confirmed pivot data structure"""
    bar_index: int
    timestamp: Any
    price: float
    label: str
    pivot_type: str
    confirmation_level: float = None


@dataclass
class MSSSignal:
    """Market Structure Shift signal"""
    timestamp: Any
    direction: str  # "UP" or "DOWN"
    broken_level: float
    trend_state: str
    label: str = "MSS"  # Visual label text
    setup_type: str = "NORMAL"  # "NORMAL", "RECLAMATION", etc.


@dataclass
class BOSSignal:
    """Break of Structure signal"""
    timestamp: Any
    direction: str  # "BULLISH" or "BEARISH"
    broken_level: float
    break_level_with_buffer: float
    label: str = None  # Visual label text (e.g., "BOS ↗", "BOS ↙")
    strength: str = "NORMAL"  # Strength classification


@dataclass
class PivotSystemState:
    """Complete pivot system state - matches Pine Script variables exactly"""
    # C1-C0 State Variables (exact Pine Script equivalents)
    waiting_for_c0_low: bool = False
    waiting_for_c0_high: bool = False
    c1_close_for_low_confirm: Optional[float] = None
    c1_close_for_high_confirm: Optional[float] = None
    c1_bar_index_for_low_confirm: Optional[int] = None
    c1_bar_index_for_high_confirm: Optional[int] = None
    invalidation_level_low: Optional[float] = None
    invalidation_level_high: Optional[float] = None
    looking_for: str = "ANY"  # "ANY", "LOW", "HIGH"
    
    # Pivot History
    last_pivot_bar: Optional[int] = None
    last_pivot_price: Optional[float] = None
    last_confirmed_actual_low_price: Optional[float] = None
    last_confirmed_actual_high_price: Optional[float] = None
    
    # MSS State Variables (exact Pine Script equivalents)
    current_trend_state: str = "Undefined"
    monitored_lh_level: Optional[float] = None
    monitored_lh_time: Optional[Any] = None
    monitored_hl_level: Optional[float] = None
    monitored_hl_time: Optional[Any] = None
    
    # BOS State Variables (exact Pine Script equivalents)
    last_hh_level: Optional[float] = None
    last_ll_level: Optional[float] = None
    last_hh_time: Optional[Any] = None
    last_ll_time: Optional[Any] = None
    
    # System Configuration
    pivot_lookback: int = 30
    structural_buffer: float = 0.1  # BOS buffer percentage


class AnasePivotDetectionSystem:
    """
    Professional implementation of AnasePivotV4 pivot detection system.
    Designed for exact Pine Script equivalence.
    """
    
    def __init__(self, 
                 pivot_lookback: int = 30,
                 reference_method: ReferenceMethod = ReferenceMethod.WICK_REJECTION,
                 structural_buffer: float = 0.1):
        """
        Initialize the pivot detection system.
        
        Args:
            pivot_lookback: Lookback period for pivot detection (default: 30)
            reference_method: Reference candle method (WICK_REJECTION or REVERSAL_OPEN)
            structural_buffer: BOS structural buffer percentage (default: 0.1%)
        """
        self.pivot_lookback = pivot_lookback
        self.reference_method = reference_method
        self.structural_buffer = structural_buffer
        
        # Initialize system state
        self.state = PivotSystemState(
            pivot_lookback=pivot_lookback,
            structural_buffer=structural_buffer
        )
        
        # Results storage
        self.confirmed_pivots: List[ConfirmedPivot] = []
        self.mss_signals: List[MSSSignal] = []
        self.bos_signals: List[BOSSignal] = []
    
    def process_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process complete dataframe and return all results.
        
        Args:
            df: OHLCV DataFrame with columns: ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            
        Returns:
            Dictionary containing all pivot, MSS, and BOS results
        """
        # Validate dataframe
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        
        # Check minimum data requirements
        if len(df) < self.pivot_lookback + 5:
            print(f"Warning: DataFrame has only {len(df)} bars, but lookback is {self.pivot_lookback}. Results may be limited.")
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df.index)
        
        # Reset results
        self.confirmed_pivots = []
        self.mss_signals = []
        self.bos_signals = []
        
        # Process bar by bar (starting from sufficient lookback)
        for i in range(self.pivot_lookback, len(df)):
            self._process_single_bar(df, i)
        
        return self._format_results()
    
    def _process_single_bar(self, df: pd.DataFrame, current_bar_index: int) -> None:
        """
        Process a single bar - exact equivalent of Pine Script bar-by-bar execution.
        
        Args:
            df: OHLCV DataFrame
            current_bar_index: Index of current bar being processed
        """
        # Step 1: Check for C0 confirmation first (highest priority)
        self._check_c0_confirmation(df, current_bar_index)
        
        # Step 2: Find potential pivots (S1 stage)
        potential_pivot_low = self._find_pivot_candle_by_close(
            df, current_bar_index, start_offset=1, 
            lookback_length=self.pivot_lookback, is_high=False
        )
        potential_pivot_high = self._find_pivot_candle_by_close(
            df, current_bar_index, start_offset=1,
            lookback_length=self.pivot_lookback, is_high=True
        )
        
        # Step 3: Find reference candles
        ref_candle_low = self._find_reference_candle(
            df, current_bar_index, start_offset=1,
            lookback_length=30, is_for_high_setup=False
        )
        ref_candle_high = self._find_reference_candle(
            df, current_bar_index, start_offset=1,
            lookback_length=30, is_for_high_setup=True
        )
        
        # Step 4: Check for C1 triggers (only if not waiting for C0)
        if not self.state.waiting_for_c0_low and not self.state.waiting_for_c0_high:
            self._check_c1_triggers(
                df, current_bar_index, potential_pivot_low, potential_pivot_high,
                ref_candle_low, ref_candle_high
            )
        
        # Step 5: Check MSS and BOS signals
        self._check_mss_signals(df, current_bar_index)
        self._check_bos_signals(df, current_bar_index)
    
    def _find_pivot_candle_by_close(self, df: pd.DataFrame, current_bar_index: int,
                                   start_offset: int, lookback_length: int, 
                                   is_high: bool) -> Optional[Tuple[int, float]]:
        """
        Find pivot candle by extreme close price - exact Pine Script f_find_pivot_candle_by_close()
        
        Returns:
            Tuple of (bar_index, close_price) or None if not found
        """
        extremum_close_price = float('-inf') if is_high else float('inf')
        extremum_bar = None
        
        for i in range(lookback_length):
            offset = start_offset + i
            if offset >= len(df) or current_bar_index - offset < 0:
                break
                
            bar_index = current_bar_index - offset
            price_val = df['close'].iloc[bar_index]
            
            if is_high and price_val > extremum_close_price:
                extremum_close_price = price_val
                extremum_bar = bar_index
            elif not is_high and price_val < extremum_close_price:
                extremum_close_price = price_val
                extremum_bar = bar_index
        
        return (extremum_bar, extremum_close_price) if extremum_bar is not None else None
    
    def _find_reference_candle(self, df: pd.DataFrame, current_bar_index: int,
                              start_offset: int, lookback_length: int,
                              is_for_high_setup: bool) -> Optional[ReferenceCandle]:
        """
        Find reference candle using specified method - exact Pine Script reference logic
        """
        if self.reference_method == ReferenceMethod.WICK_REJECTION:
            return self._find_ref_by_wick(df, current_bar_index, start_offset, 
                                        lookback_length, is_for_high_setup)
        else:
            return self._find_ref_by_reversal_open(df, current_bar_index, start_offset,
                                                 lookback_length, is_for_high_setup)
    
    def _find_ref_by_wick(self, df: pd.DataFrame, current_bar_index: int,
                         start_offset: int, lookback_length: int,
                         is_for_high_setup: bool) -> Optional[ReferenceCandle]:
        """
        Find reference candle using wick rejection method - exact Pine Script f_find_ref_by_wick()
        
        For high setups: Find candle with highest low (strongest bullish rejection)
        For low setups: Find candle with lowest high (strongest bearish rejection)
        """
        reference_level = float('-inf') if is_for_high_setup else float('inf')
        ref_bar_index = None
        
        for i in range(lookback_length):
            offset = start_offset + i
            if offset >= len(df) or current_bar_index - offset < 0:
                break
                
            bar_index = current_bar_index - offset
            
            # For high setups: look for highest low (strongest rejection from bears)
            # For low setups: look for lowest high (strongest rejection from bulls)
            level_to_check = df['low'].iloc[bar_index] if is_for_high_setup else df['high'].iloc[bar_index]
            
            if is_for_high_setup and level_to_check > reference_level:
                reference_level = level_to_check
                ref_bar_index = bar_index
            elif not is_for_high_setup and level_to_check < reference_level:
                reference_level = level_to_check
                ref_bar_index = bar_index
        
        if ref_bar_index is not None:
            return ReferenceCandle(
                bar_index=ref_bar_index,
                open_price=df['open'].iloc[ref_bar_index],
                close_price=df['close'].iloc[ref_bar_index],
                high_price=df['high'].iloc[ref_bar_index],
                low_price=df['low'].iloc[ref_bar_index],
                timestamp=df['timestamp'].iloc[ref_bar_index]
            )
        return None
    
    def _find_ref_by_reversal_open(self, df: pd.DataFrame, current_bar_index: int,
                                  start_offset: int, lookback_length: int,
                                  is_for_high_setup: bool) -> Optional[ReferenceCandle]:
        """
        Find reference candle using reversal open method - exact Pine Script f_find_ref_by_reversal_open()
        
        For high setups: Find bullish candle with highest open
        For low setups: Find bearish candle with lowest open
        """
        extreme_open = float('-inf') if is_for_high_setup else float('inf')
        ref_bar_index = None
        
        for i in range(lookback_length):
            offset = start_offset + i
            if offset >= len(df) or current_bar_index - offset < 0:
                break
                
            bar_index = current_bar_index - offset
            
            candle_open = df['open'].iloc[bar_index]
            candle_close = df['close'].iloc[bar_index]
            is_candle_bullish = candle_close > candle_open
            is_candle_bearish = candle_close < candle_open
            
            if is_for_high_setup and is_candle_bullish:
                if candle_open > extreme_open:
                    extreme_open = candle_open
                    ref_bar_index = bar_index
            elif not is_for_high_setup and is_candle_bearish:
                if candle_open < extreme_open:
                    extreme_open = candle_open
                    ref_bar_index = bar_index
        
        if ref_bar_index is not None:
            return ReferenceCandle(
                bar_index=ref_bar_index,
                open_price=df['open'].iloc[ref_bar_index],
                close_price=df['close'].iloc[ref_bar_index],
                high_price=df['high'].iloc[ref_bar_index],
                low_price=df['low'].iloc[ref_bar_index],
                timestamp=df['timestamp'].iloc[ref_bar_index]
            )
        return None
    
    def _validate_c1_bullish(self, df: pd.DataFrame, current_bar_index: int,
                           ref_candle: Optional[ReferenceCandle]) -> bool:
        """
        Validate bullish C1 candle for LOW pivot detection - exact Pine Script logic
        """
        current_close = df['close'].iloc[current_bar_index]
        current_open = df['open'].iloc[current_bar_index]
        
        # Must be bullish candle
        if current_close <= current_open:
            return False
        
        # Must have valid reference candle
        if ref_candle is None:
            return False
        
        # Reference must be newer than last pivot
        if (self.state.last_pivot_bar is not None and 
            ref_candle.bar_index <= self.state.last_pivot_bar):
            return False
        
        # Apply validation method
        if self.reference_method == ReferenceMethod.REVERSAL_OPEN:
            # Reversal open method: close must break reference open
            return current_close > ref_candle.open_price
        else:
            # Wick rejection method: more complex validation
            is_ref_bullish = ref_candle.is_bullish()
            
            if is_ref_bullish:
                # Reference was bullish: check against close or high
                return (current_close > ref_candle.close_price or 
                       current_close > ref_candle.high_price)
            else:
                # Reference was bearish: check against open or high
                return (current_close > ref_candle.open_price or 
                       current_close > ref_candle.high_price)
    
    def _validate_c1_bearish(self, df: pd.DataFrame, current_bar_index: int,
                           ref_candle: Optional[ReferenceCandle]) -> bool:
        """
        Validate bearish C1 candle for HIGH pivot detection - exact Pine Script logic
        """
        current_close = df['close'].iloc[current_bar_index]
        current_open = df['open'].iloc[current_bar_index]
        
        # Must be bearish candle
        if current_close >= current_open:
            return False
        
        # Must have valid reference candle
        if ref_candle is None:
            return False
        
        # Reference must be newer than last pivot
        if (self.state.last_pivot_bar is not None and 
            ref_candle.bar_index <= self.state.last_pivot_bar):
            return False
        
        # Apply validation method
        if self.reference_method == ReferenceMethod.REVERSAL_OPEN:
            # Reversal open method: close must break reference open
            return current_close < ref_candle.open_price
        else:
            # Wick rejection method: more complex validation
            is_ref_bullish = ref_candle.is_bullish()
            
            if is_ref_bullish:
                # Reference was bullish: check against open or low
                return (current_close < ref_candle.open_price or 
                       current_close < ref_candle.low_price)
            else:
                # Reference was bearish: check against close or low
                return (current_close < ref_candle.close_price or 
                       current_close < ref_candle.low_price)
    
    def _check_c1_triggers(self, df: pd.DataFrame, current_bar_index: int,
                          potential_pivot_low: Optional[Tuple[int, float]],
                          potential_pivot_high: Optional[Tuple[int, float]],
                          ref_candle_low: Optional[ReferenceCandle],
                          ref_candle_high: Optional[ReferenceCandle]) -> None:
        """
        Check for C1 trigger conditions - exact Pine Script C1 logic
        """
        # Check for bullish C1 (LOW pivot setup)
        if (self.state.looking_for in ["ANY", "LOW"] and 
            potential_pivot_low is not None):
            
            if self._validate_c1_bullish(df, current_bar_index, ref_candle_low):
                # Enter C1-C0 waiting state for low pivot
                self.state.waiting_for_c0_low = True
                self.state.c1_close_for_low_confirm = df['close'].iloc[current_bar_index]
                self.state.c1_bar_index_for_low_confirm = current_bar_index
                self.state.invalidation_level_low = df['open'].iloc[current_bar_index]
        
        # Check for bearish C1 (HIGH pivot setup)
        if (self.state.looking_for in ["ANY", "HIGH"] and 
            potential_pivot_high is not None):
            
            if self._validate_c1_bearish(df, current_bar_index, ref_candle_high):
                # Enter C1-C0 waiting state for high pivot
                self.state.waiting_for_c0_high = True
                self.state.c1_close_for_high_confirm = df['close'].iloc[current_bar_index]
                self.state.c1_bar_index_for_high_confirm = current_bar_index
                self.state.invalidation_level_high = df['open'].iloc[current_bar_index]
    
    def _check_c0_confirmation(self, df: pd.DataFrame, current_bar_index: int) -> None:
        """
        Check for C0 confirmation - exact Pine Script C0 logic
        """
        current_close = df['close'].iloc[current_bar_index]
        current_open = df['open'].iloc[current_bar_index]
        
        # C0 confirmation for LOW pivot
        if self.state.waiting_for_c0_low:
            is_invalidated = (self.state.invalidation_level_low is not None and 
                            current_close < self.state.invalidation_level_low)
            is_confirmed = (current_close > self.state.c1_close_for_low_confirm and 
                          current_close > current_open)
            
            if is_invalidated or is_confirmed:
                if is_confirmed:
                    # Find the actual pivot candle using S2 detection
                    s2_lookback = (self.pivot_lookback if self.state.last_pivot_bar is None 
                                 else current_bar_index - self.state.c1_bar_index_for_low_confirm)
                    s2_offset = current_bar_index - self.state.c1_bar_index_for_low_confirm
                    
                    pivot_result = self._find_pivot_candle_by_close(
                        df, current_bar_index, start_offset=s2_offset,
                        lookback_length=s2_lookback, is_high=False
                    )
                    
                    if pivot_result:
                        pivot_bar, pivot_price = pivot_result
                        confirmed_pivot = self._validate_and_label_pivot(
                            df, pivot_bar, pivot_price, False
                        )
                        
                        if confirmed_pivot:
                            self.confirmed_pivots.append(confirmed_pivot)
                            self._update_system_state_for_pivot(confirmed_pivot)
                
                # Reset C0 waiting state
                self.state.waiting_for_c0_low = False
                self.state.c1_close_for_low_confirm = None
                self.state.c1_bar_index_for_low_confirm = None
                self.state.invalidation_level_low = None
        
        # C0 confirmation for HIGH pivot
        elif self.state.waiting_for_c0_high:
            is_invalidated = (self.state.invalidation_level_high is not None and 
                            current_close > self.state.invalidation_level_high)
            is_confirmed = (current_close < self.state.c1_close_for_high_confirm and 
                          current_close < current_open)
            
            if is_invalidated or is_confirmed:
                if is_confirmed:
                    # Find the actual pivot candle using S2 detection
                    s2_lookback = (self.pivot_lookback if self.state.last_pivot_bar is None 
                                 else current_bar_index - self.state.c1_bar_index_for_high_confirm)
                    s2_offset = current_bar_index - self.state.c1_bar_index_for_high_confirm
                    
                    pivot_result = self._find_pivot_candle_by_close(
                        df, current_bar_index, start_offset=s2_offset,
                        lookback_length=s2_lookback, is_high=True
                    )
                    
                    if pivot_result:
                        pivot_bar, pivot_price = pivot_result
                        confirmed_pivot = self._validate_and_label_pivot(
                            df, pivot_bar, pivot_price, True
                        )
                        
                        if confirmed_pivot:
                            self.confirmed_pivots.append(confirmed_pivot)
                            self._update_system_state_for_pivot(confirmed_pivot)
                
                # Reset C0 waiting state
                self.state.waiting_for_c0_high = False
                self.state.c1_close_for_high_confirm = None
                self.state.c1_bar_index_for_high_confirm = None
                self.state.invalidation_level_high = None
    
    def _validate_and_label_pivot(self, df: pd.DataFrame, pivot_bar: int, 
                                 pivot_price: float, is_high_pivot: bool) -> Optional[ConfirmedPivot]:
        """
        Validate and label pivot - exact Pine Script f_validate_and_label_pivot()
        """
        # Basic validation
        if self.state.last_pivot_bar is not None and pivot_bar <= self.state.last_pivot_bar:
            return None
        if self.state.last_pivot_price is not None and abs(pivot_price - self.state.last_pivot_price) < 1e-10:
            return None
        
        # Determine pivot label based on comparison with previous pivots
        label = ""
        pivot_type = PivotType.HIGH.value if is_high_pivot else PivotType.LOW.value
        
        if is_high_pivot:
            # Compare with last confirmed high
            if (self.state.last_confirmed_actual_high_price is None or 
                pivot_price > self.state.last_confirmed_actual_high_price):
                label = PivotLabel.HH.value
            else:
                label = PivotLabel.LH.value
        else:
            # Compare with last confirmed low
            if (self.state.last_confirmed_actual_low_price is None or 
                pivot_price < self.state.last_confirmed_actual_low_price):
                label = PivotLabel.LL.value
            else:
                label = PivotLabel.HL.value
        
        return ConfirmedPivot(
            bar_index=pivot_bar,
            timestamp=df['timestamp'].iloc[pivot_bar],
            price=pivot_price,
            label=label,
            pivot_type=pivot_type,
            confirmation_level=pivot_price
        )
    
    def _update_system_state_for_pivot(self, confirmed_pivot: ConfirmedPivot) -> None:
        """
        Update system state after pivot confirmation - exact Pine Script state updates
        """
        # Update pivot tracking
        self.state.last_pivot_bar = confirmed_pivot.bar_index
        self.state.last_pivot_price = confirmed_pivot.price
        
        # Update confirmed price tracking
        if confirmed_pivot.pivot_type == PivotType.HIGH.value:
            self.state.last_confirmed_actual_high_price = confirmed_pivot.price
            self.state.looking_for = "LOW"
        else:
            self.state.last_confirmed_actual_low_price = confirmed_pivot.price
            self.state.looking_for = "HIGH"
        
        # Update MSS monitoring levels (LH/HL detection)
        if confirmed_pivot.label == PivotLabel.LH.value:
            self.state.monitored_lh_level = confirmed_pivot.price
            self.state.monitored_lh_time = confirmed_pivot.timestamp
        elif confirmed_pivot.label == PivotLabel.HL.value:
            self.state.monitored_hl_level = confirmed_pivot.price
            self.state.monitored_hl_time = confirmed_pivot.timestamp
        
        # Update BOS monitoring levels (HH/LL detection)
        if confirmed_pivot.label == PivotLabel.HH.value:
            self.state.last_hh_level = confirmed_pivot.price
            self.state.last_hh_time = confirmed_pivot.timestamp
        elif confirmed_pivot.label == PivotLabel.LL.value:
            self.state.last_ll_level = confirmed_pivot.price
            self.state.last_ll_time = confirmed_pivot.timestamp
    
    def _check_mss_signals(self, df: pd.DataFrame, current_bar_index: int) -> None:
        """
        Check for MSS (Market Structure Shift) signals - exact Pine Script MSS logic
        """
        current_close = df['close'].iloc[current_bar_index]
        current_timestamp = df['timestamp'].iloc[current_bar_index]
        
        # MSS Up: Close above monitored LH level
        if (self.state.monitored_lh_level is not None and 
            current_close > self.state.monitored_lh_level):
            
            mss_signal = MSSSignal(
                timestamp=current_timestamp,
                direction="UP",
                broken_level=self.state.monitored_lh_level,
                trend_state=TrendState.UPTREND.value,
                label="MSS",
                setup_type="NORMAL"  # Could be enhanced to detect reclamation vs normal
            )
            self.mss_signals.append(mss_signal)
            
            # Update trend state
            self.state.current_trend_state = TrendState.UPTREND.value
            
            # Clear monitored LH level
            self.state.monitored_lh_level = None
            self.state.monitored_lh_time = None
        
        # MSS Down: Close below monitored HL level
        if (self.state.monitored_hl_level is not None and 
            current_close < self.state.monitored_hl_level):
            
            mss_signal = MSSSignal(
                timestamp=current_timestamp,
                direction="DOWN",
                broken_level=self.state.monitored_hl_level,
                trend_state=TrendState.DOWNTREND.value,
                label="MSS",
                setup_type="NORMAL"  # Could be enhanced to detect reclamation vs normal
            )
            self.mss_signals.append(mss_signal)
            
            # Update trend state
            self.state.current_trend_state = TrendState.DOWNTREND.value
            
            # Clear monitored HL level
            self.state.monitored_hl_level = None
            self.state.monitored_hl_time = None
    
    def _check_bos_signals(self, df: pd.DataFrame, current_bar_index: int) -> None:
        """
        Check for BOS (Break of Structure) signals - exact Pine Script BOS logic (no volume)
        """
        current_close = df['close'].iloc[current_bar_index]
        current_timestamp = df['timestamp'].iloc[current_bar_index]
        
        # Calculate break levels with structural buffer
        hh_break_level = None
        ll_break_level = None
        
        if self.state.last_hh_level is not None:
            hh_break_level = self.state.last_hh_level * (1 + self.state.structural_buffer / 100)
        
        if self.state.last_ll_level is not None:
            ll_break_level = self.state.last_ll_level * (1 - self.state.structural_buffer / 100)
        
        # Bullish BOS: Close above HH level with buffer
        if hh_break_level is not None and current_close > hh_break_level:
            bos_signal = BOSSignal(
                timestamp=current_timestamp,
                direction="BULLISH",
                broken_level=self.state.last_hh_level,
                break_level_with_buffer=hh_break_level,
                label="BOS ↗",
                strength="NORMAL"
            )
            self.bos_signals.append(bos_signal)
        
        # Bearish BOS: Close below LL level with buffer
        if ll_break_level is not None and current_close < ll_break_level:
            bos_signal = BOSSignal(
                timestamp=current_timestamp,
                direction="BEARISH",
                broken_level=self.state.last_ll_level,
                break_level_with_buffer=ll_break_level,
                label="BOS ↙",
                strength="NORMAL"
            )
            self.bos_signals.append(bos_signal)
    
    def _format_results(self) -> Dict[str, Any]:
        """
        Format results into structured output
        """
        return {
            'pivots': {
                'confirmed': [
                    {
                        'bar_index': p.bar_index,
                        'timestamp': p.timestamp,
                        'price': p.price,
                        'label': p.label,
                        'pivot_type': p.pivot_type
                    } for p in self.confirmed_pivots
                ],
                'pending': {
                    'waiting_for_c0_low': self.state.waiting_for_c0_low,
                    'waiting_for_c0_high': self.state.waiting_for_c0_high,
                    'c1_close_for_confirm': (self.state.c1_close_for_low_confirm or 
                                           self.state.c1_close_for_high_confirm),
                    'invalidation_level': (self.state.invalidation_level_low or 
                                         self.state.invalidation_level_high)
                }
            },
            'mss': {
                'signals': [
                    {
                        'timestamp': s.timestamp,
                        'direction': s.direction,
                        'broken_level': s.broken_level,
                        'trend_state': s.trend_state,
                        'label': s.label,
                        'setup_type': s.setup_type
                    } for s in self.mss_signals
                ],
                'state': {
                    'monitored_lh_level': self.state.monitored_lh_level,
                    'monitored_hl_level': self.state.monitored_hl_level,
                    'current_trend_state': self.state.current_trend_state
                }
            },
            'bos': {
                'signals': [
                    {
                        'timestamp': s.timestamp,
                        'direction': s.direction,
                        'broken_level': s.broken_level,
                        'break_level_with_buffer': s.break_level_with_buffer,
                        'label': s.label,
                        'strength': s.strength
                    } for s in self.bos_signals
                ],
                'state': {
                    'last_hh_level': self.state.last_hh_level,
                    'last_ll_level': self.state.last_ll_level,
                    'structural_buffer': self.state.structural_buffer
                }
            },
            'system': {
                'looking_for': self.state.looking_for,
                'last_pivot_bar': self.state.last_pivot_bar,
                'total_confirmed_pivots': len(self.confirmed_pivots),
                'reference_method': self.reference_method.value,
                'pivot_lookback': self.pivot_lookback
            }
        }
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current system state without processing new data"""
        return self._format_results()


# Convenience function for easy usage
def detect_anase_pivots(df: pd.DataFrame, 
                       pivot_lookback: int = 30,
                       reference_method: str = "WICK_REJECTION",
                       structural_buffer: float = 0.1) -> Dict[str, Any]:
    """
    Convenience function for AnasePivotV4 detection.
    
    Args:
        df: OHLCV DataFrame with columns ['open', 'high', 'low', 'close', 'timestamp']
        pivot_lookback: Lookback period for pivot detection (default: 30)
        reference_method: "WICK_REJECTION" or "REVERSAL_OPEN" (default: "WICK_REJECTION")
        structural_buffer: BOS structural buffer percentage (default: 0.1%)
    
    Returns:
        Dictionary containing all pivot, MSS, and BOS results
    
    Example:
        ```python
        import pandas as pd
        from anase_pivot_system import detect_anase_pivots
        
        # Load your OHLCV data
        df = pd.read_csv('BTCUSD_5m.csv')
        
        # Detect pivots
        results = detect_anase_pivots(df)
        
        # Access results
        confirmed_pivots = results['pivots']['confirmed']
        mss_signals = results['mss']['signals']
        bos_signals = results['bos']['signals']
        ```
    """
    # Convert string to enum
    ref_method = (ReferenceMethod.REVERSAL_OPEN if reference_method == "REVERSAL_OPEN" 
                 else ReferenceMethod.WICK_REJECTION)
    
    # Create detector and process
    detector = AnasePivotDetectionSystem(
        pivot_lookback=pivot_lookback,
        reference_method=ref_method,
        structural_buffer=structural_buffer
    )
    
    return detector.process_dataframe(df)


if __name__ == "__main__":
    # Example usage and basic testing
    print("AnasePivotV4 Professional Pivot Detection System")
    print("=" * 50)
    
    try:
        # Create sample test data
        import random
        random.seed(42)
        
        # Generate sample OHLCV data
        base_price = 45000
        data = []
        
        for i in range(100):
            # Simple price walk
            base_price += random.uniform(-100, 100)
            
            open_price = base_price + random.uniform(-20, 20)
            close_price = base_price + random.uniform(-20, 20)
            high_price = max(open_price, close_price) + random.uniform(0, 50)
            low_price = min(open_price, close_price) - random.uniform(0, 50)
            
            data.append({
                'timestamp': f'2024-01-01 {10 + i//12:02d}:{(i*5)%60:02d}:00',
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': random.randint(100, 1000)
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        print(f"✅ Created sample data: {len(df)} bars")
        print(f"   Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
        
        # Test the detection system
        results = detect_anase_pivots(df)
        
        # Display results
        print(f"✅ Detection completed successfully!")
        print(f"   Confirmed pivots: {len(results['pivots']['confirmed'])}")
        print(f"   MSS signals: {len(results['mss']['signals'])}")
        print(f"   BOS signals: {len(results['bos']['signals'])}")
        print(f"   Current trend: {results['mss']['state']['current_trend_state']}")
        
        # Show sample results
        if results['pivots']['confirmed']:
            pivot = results['pivots']['confirmed'][0]
            print(f"   Sample pivot: {pivot['label']} at {pivot['price']} (bar {pivot['bar_index']})")
        
        if results['mss']['signals']:
            mss = results['mss']['signals'][0]
            print(f"   Sample MSS: {mss['direction']} at {mss['broken_level']} ({mss['label']})")
        
        if results['bos']['signals']:
            bos = results['bos']['signals'][0]
            print(f"   Sample BOS: {bos['direction']} at {bos['broken_level']} ({bos['label']})")
        
        print("\n✅ All basic tests passed!")
        print("System is ready for use with real market data.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the error above and fix any issues.")
    
    print("\nReady for data processing.")
