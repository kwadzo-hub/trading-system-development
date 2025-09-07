# AnasePivot Core System - Professional Python Implementation
# Exact replication of Pine Script Oryon indicator logic

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum
import math
import logging
from copy import deepcopy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================================================================================
# ENUMS AND CONSTANTS
# ==================================================================================

class PivotType(Enum):
    HIGH = "HIGH"
    LOW = "LOW"

class PivotLabel(Enum):
    HH = "HH"  # Higher High
    LH = "LH"  # Lower High  
    HL = "HL"  # Higher Low
    LL = "LL"  # Lower Low

class ReferenceMethod(Enum):
    WICK_REJECTION = "WICK_REJECTION"
    REVERSAL_OPEN = "REVERSAL_OPEN"

class LookingFor(Enum):
    LOW = "LOW"
    HIGH = "HIGH" 
    ANY = "ANY"

class SignalType(Enum):
    BULLISH_C1 = "BULLISH_C1"
    BEARISH_C1 = "BEARISH_C1"


  # ==================================================================================
# DATA STRUCTURES
# ==================================================================================

@dataclass
class Candle:
    """Represents a single candlestick with OHLC data"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    bar_index: int
    
    def __post_init__(self):
        """Validate candle data"""
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError(f"Invalid candle data: high={self.high}, low={self.low}, open={self.open}, close={self.close}")
        if any(x <= 0 for x in [self.open, self.high, self.low, self.close]):
            raise ValueError("All price values must be positive")
    
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    def is_bearish(self) -> bool:
        return self.close < self.open
    
    def body_size(self) -> float:
        return abs(self.close - self.open)
    
    def upper_wick_size(self) -> float:
        return self.high - max(self.open, self.close)
    
    def lower_wick_size(self) -> float:
        return min(self.open, self.close) - self.low

@dataclass
class PivotData:
    """Represents a detected pivot point"""
    bar_index: int
    timestamp: int
    price_level: float
    pivot_type: PivotType
    label: Optional[PivotLabel] = None
    confirmed: bool = False
    
    def __post_init__(self):
        if self.price_level <= 0:
            raise ValueError("Price level must be positive")

@dataclass
class ReferenceCandle:
    """Represents a reference candle for C1 detection"""
    bar_index: int
    open: float
    high: float
    low: float
    close: float
    method: ReferenceMethod
    is_bullish: bool
    
    def __post_init__(self):
        if any(x <= 0 for x in [self.open, self.high, self.low, self.close]):
            raise ValueError("All price values must be positive")

@dataclass
class C1Signal:
    """Represents a C1 detection signal"""
    detected: bool
    signal_type: SignalType
    trigger_price: float
    invalidation_level: float
    reference_candle: ReferenceCandle
    timestamp: int
    bar_index: int
  @dataclass
class StateManager:
    """Manages the C0-C1 state machine and pivot tracking"""
    # C1-C0 State Machine
    waiting_for_c0_low: bool = False
    waiting_for_c0_high: bool = False
    c1_close_for_low_confirm: Optional[float] = None
    c1_close_for_high_confirm: Optional[float] = None
    c1_bar_index_for_low_confirm: Optional[int] = None
    c1_bar_index_for_high_confirm: Optional[int] = None
    invalidation_level_low: Optional[float] = None
    invalidation_level_high: Optional[float] = None
    
    # Pivot Tracking
    last_pivot_bar: Optional[int] = None
    last_confirmed_high_price: Optional[float] = None
    last_confirmed_low_price: Optional[float] = None
    looking_for: LookingFor = LookingFor.ANY
    
    # MSS/BOS Tracking
    last_bos_high: Optional[float] = None
    last_bos_low: Optional[float] = None
    
    def reset_c1_state(self):
        """Reset C1 state variables"""
        self.waiting_for_c0_low = False
        self.waiting_for_c0_high = False
        self.c1_close_for_low_confirm = None
        self.c1_close_for_high_confirm = None
        self.c1_bar_index_for_low_confirm = None
        self.c1_bar_index_for_high_confirm = None
        self.invalidation_level_low = None
        self.invalidation_level_high = None

@dataclass
class PivotSystemConfig:
    """Configuration for the pivot system"""
    pivot_lookback: int = 30
    reference_method: ReferenceMethod = ReferenceMethod.WICK_REJECTION
    max_bars_back: int = 5000
    enable_c1_detection: bool = True
    enable_mss_bos_detection: bool = True
    max_pivots_stored: int = 100
    min_price_level: float = 1e-10
    max_price_level: float = 1e10
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if not (5 <= self.pivot_lookback <= 100):
            raise ValueError("pivot_lookback must be between 5 and 100")
        if self.max_bars_back < 1000:
            raise ValueError("max_bars_back must be at least 1000")
        if self.max_pivots_stored < 10:
            raise ValueError("max_pivots_stored must be at least 10")
        return True

# ==================================================================================
# UTILITY FUNCTIONS
# ==================================================================================

class ValidationUtils:
    """Utility functions for data validation"""
    
    @staticmethod
    def is_valid_price(price: float) -> bool:
        """Check if price is valid (positive and finite)"""
        return price > 0 and math.isfinite(price)
    
    @staticmethod
    def is_valid_bar_index(bar_index: int) -> bool:
        """Check if bar index is valid"""
        return isinstance(bar_index, int) and bar_index >= 0
    
    @staticmethod
    def is_valid_lookback(lookback: int, min_val: int = 5, max_val: int = 100) -> bool:
        """Check if lookback period is valid"""
        return isinstance(lookback, int) and min_val <= lookback <= max_val


      # ==================================================================================
# PIVOT DETECTION ENGINE
# ==================================================================================

class PivotDetectionEngine:
    """Handles core pivot detection logic - exact Pine Script replication"""
    
    def __init__(self, config: PivotSystemConfig):
        self.config = config
        
    def find_pivot_candle_by_close(self, candle_data: List[Candle], offset: int, 
                                 lookback: int, is_high: bool) -> Tuple[Optional[int], Optional[float]]:
        """
        Exact implementation of Pine Script f_find_pivot_candle_by_close
        
        Args:
            candle_data: List of candle data
            offset: Starting offset from current bar
            lookback: Number of bars to look back
            is_high: True for high pivots, False for low pivots
            
        Returns:
            Tuple of (bar_index, price_level) or (None, None) if not found
        """
        if not candle_data or len(candle_data) < offset + 1:
            return None, None
            
        if not ValidationUtils.is_valid_lookback(lookback, 1, len(candle_data)):
            return None, None
            
        current_index = len(candle_data) - 1 - offset
        if current_index < 0:
            return None, None
            
        extreme_price = float('-inf') if is_high else float('inf')
        extreme_bar_index = None
        
        # Look back from current position
        for i in range(lookback):
            check_index = current_index - i
            if check_index < 0:
                break
                
            candle = candle_data[check_index]
            check_price = candle.close
            
            if not ValidationUtils.is_valid_price(check_price):
                continue
                
            if is_high and check_price > extreme_price:
                extreme_price = check_price
                extreme_bar_index = candle.bar_index
            elif not is_high and check_price < extreme_price:
                extreme_price = check_price
                extreme_bar_index = candle.bar_index
                
        return extreme_bar_index, extreme_price if extreme_bar_index is not None else None
    
    def validate_and_label_pivot(self, candle_data: List[Candle], pivot_bar_index: int, 
                               is_high: bool, pivot_history: List[PivotData]) -> Tuple[bool, Optional[PivotLabel], float]:
        """
        Exact implementation of Pine Script f_validate_and_label_pivot
        
        Validates pivot and assigns HH/LH/HL/LL labels based on previous pivots
        """
        # Find the candle at pivot_bar_index
        pivot_candle = None
        for candle in candle_data:
            if candle.bar_index == pivot_bar_index:
                pivot_candle = candle
                break
                
        if not pivot_candle:
            return False, None, 0.0
            
        pivot_price = pivot_candle.high if is_high else pivot_candle.low
        
        if not ValidationUtils.is_valid_price(pivot_price):
            return False, None, 0.0
            
        # Find the last pivot of the same type
        last_same_type_pivot = None
        for pivot in reversed(pivot_history):
            if pivot.confirmed and pivot.pivot_type == (PivotType.HIGH if is_high else PivotType.LOW):
                last_same_type_pivot = pivot
                break
                
        # Determine label based on comparison with last same-type pivot
        if last_same_type_pivot is None:
            # First pivot of this type
            label = PivotLabel.HH if is_high else PivotLabel.HL
        else:
            if is_high:
                # High pivot labeling
                label = PivotLabel.HH if pivot_price > last_same_type_pivot.price_level else PivotLabel.LH
            else:
                # Low pivot labeling  
                label = PivotLabel.HL if pivot_price > last_same_type_pivot.price_level else PivotLabel.LL
                
        return True, label, pivot_price

# ==================================================================================
# REFERENCE CANDLE FINDER
# ==================================================================================

class ReferenceCandleFinder:
    """Implements both reference candle detection methods"""
    
    def __init__(self, config: PivotSystemConfig):
        self.config = config
        
    def find_ref_by_wick(self, candle_data: List[Candle], start_offset: int, 
                        lookback: int, is_for_high: bool) -> Optional[ReferenceCandle]:
        """
        Exact implementation of Pine Script f_find_ref_by_wick
        
        For high setups: Find candle with highest low in lookback period
        For low setups: Find candle with lowest high in lookback period
        """
        if not candle_data or len(candle_data) < start_offset + 1:
            return None
            
        current_index = len(candle_data) - 1 - start_offset
        if current_index < 0:
            return None
            
        extreme_level = float('-inf') if is_for_high else float('inf')
        reference_candle = None
        
        for i in range(lookback):
            check_index = current_index - i
            if check_index < 0:
                break
                
            candle = candle_data[check_index]
            level_to_check = candle.low if is_for_high else candle.high
            
            if not ValidationUtils.is_valid_price(level_to_check):
                continue
                
            if is_for_high and level_to_check > extreme_level:
                extreme_level = level_to_check
                reference_candle = candle
            elif not is_for_high and level_to_check < extreme_level:
                extreme_level = level_to_check
                reference_candle = candle
                
        if reference_candle:
            return ReferenceCandle(
                bar_index=reference_candle.bar_index,
                open=reference_candle.open,
                high=reference_candle.high,
                low=reference_candle.low,
                close=reference_candle.close,
                method=ReferenceMethod.WICK_REJECTION,
                is_bullish=reference_candle.is_bullish()
            )
            
        return None
        
    def find_ref_by_reversal_open(self, candle_data: List[Candle], start_offset: int,
                                lookback: int, is_for_high: bool) -> Optional[ReferenceCandle]:
        """
        Exact implementation of Pine Script f_find_ref_by_reversal_open
        
        For high setups: Find highest open among bullish candles
        For low setups: Find lowest open among bearish candles
        """
        if not candle_data or len(candle_data) < start_offset + 1:
            return None
            
        current_index = len(candle_data) - 1 - start_offset
        if current_index < 0:
            return None
            
        extreme_open = float('-inf') if is_for_high else float('inf')
        reference_candle = None
        
        for i in range(lookback):
            check_index = current_index - i
            if check_index < 0:
                break
                
            candle = candle_data[check_index]
            
            if not ValidationUtils.is_valid_price(candle.open):
                continue
                
            # For high setups: look for bullish candles with highest open
            # For low setups: look for bearish candles with lowest open
            if is_for_high and candle.is_bullish():
                if candle.open > extreme_open:
                    extreme_open = candle.open
                    reference_candle = candle
            elif not is_for_high and candle.is_bearish():
                if candle.open < extreme_open:
                    extreme_open = candle.open
                    reference_candle = candle
                    
        if reference_candle:
            return ReferenceCandle(
                bar_index=reference_candle.bar_index,
                open=reference_candle.open,
                high=reference_candle.high,
                low=reference_candle.low,
                close=reference_candle.close,
                method=ReferenceMethod.REVERSAL_OPEN,
                is_bullish=reference_candle.is_bullish()
            )
            
        return None
        
    def get_safe_lookback(self, pivot_lookback: int, last_pivot_bar: Optional[int], 
                         current_bar: int) -> int:
        """
        Exact implementation of Pine Script f_get_safe_lookback
        
        Calculates safe lookback period to avoid overlapping with previous pivots
        """
        if last_pivot_bar is None:
            return pivot_lookback
            
        # Calculate distance from current bar to last pivot
        distance_to_last_pivot = current_bar - last_pivot_bar
        
        # Use minimum of default lookback and distance to last pivot
        return min(pivot_lookback, max(5, distance_to_last_pivot))

# ==================================================================================
# C1 DETECTION ENGINE
# ==================================================================================

class C1DetectionEngine:
    """Handles C1 candle detection and validation"""
    
    def __init__(self, config: PivotSystemConfig):
        self.config = config
        
    def detect_bullish_c1(self, current_candle: Candle, ref_candle: Optional[ReferenceCandle],
                         pivot_data: Optional[PivotData], state: StateManager) -> bool:
        """
        Exact implementation of Pine Script bullish C1 detection logic
        
        Conditions:
        1. Current candle must be bullish (close > open)
        2. Must have valid reference candle
        3. Reference must be newer than last pivot
        4. Must have pending pivot low
        5. Close must break above reference levels based on method
        """
        # Condition 1: Current candle must be bullish
        if not current_candle.is_bullish():
            return False
            
        # Condition 2: Must have valid reference candle
        if not ref_candle:
            return False
            
        # Condition 3: Reference must be newer than last pivot
        if state.last_pivot_bar is not None and ref_candle.bar_index <= state.last_pivot_bar:
            return False
            
        # Condition 4: Must have pending pivot (this is handled by the main system)
        # We assume if we're called, there's a valid pivot to confirm
        
        # Condition 5: Apply breakout logic based on method
        if self.config.reference_method == ReferenceMethod.REVERSAL_OPEN:
            # Method 2: Close must break above reference open
            return current_candle.close > ref_candle.open
        else:
            # Method 1: More complex wick rejection logic
            if ref_candle.is_bullish:
                # Reference is bullish: break above close OR high
                return (current_candle.close > ref_candle.close or 
                       current_candle.close > ref_candle.high)
            else:
                # Reference is bearish: break above open OR high
                return (current_candle.close > ref_candle.open or 
                       current_candle.close > ref_candle.high)
                       
    def detect_bearish_c1(self, current_candle: Candle, ref_candle: Optional[ReferenceCandle],
                          pivot_data: Optional[PivotData], state: StateManager) -> bool:
        """
        Exact implementation of Pine Script bearish C1 detection logic
        
        Conditions:
        1. Current candle must be bearish (close < open)
        2. Must have valid reference candle  
        3. Reference must be newer than last pivot
        4. Must have pending pivot high
        5. Close must break below reference levels based on method
        """
        # Condition 1: Current candle must be bearish
        if not current_candle.is_bearish():
            return False
            
        # Condition 2: Must have valid reference candle
        if not ref_candle:
            return False
            
        # Condition 3: Reference must be newer than last pivot
        if state.last_pivot_bar is not None and ref_candle.bar_index <= state.last_pivot_bar:
            return False
            
        # Condition 4: Must have pending pivot (handled by main system)
        
        # Condition 5: Apply breakdown logic based on method
        if self.config.reference_method == ReferenceMethod.REVERSAL_OPEN:
            # Method 2: Close must break below reference open
            return current_candle.close < ref_candle.open
        else:
            # Method 1: More complex wick rejection logic
            if ref_candle.is_bullish:
                # Reference is bullish: break below open OR low
                return (current_candle.close < ref_candle.open or 
                       current_candle.close < ref_candle.low)
            else:
                # Reference is bearish: break below close OR low
                return (current_candle.close < ref_candle.close or 
                       current_candle.close < ref_candle.low)

# ==================================================================================
# C0 CONFIRMATION ENGINE
# ==================================================================================

class C0ConfirmationEngine:
    """Handles C0 confirmation and pivot finalization"""
    
    def __init__(self, config: PivotSystemConfig):
        self.config = config
        
    def check_c0_low_confirmation(self, current_candle: Candle, state: StateManager) -> Dict:
        """
        Exact implementation of Pine Script C0 low confirmation logic
        
        Returns:
            Dict with confirmation status, invalidation status, and details
        """
        result = {
            'confirmed': False,
            'invalidated': False,
            'should_reset': False,
            'details': {}
        }
        
        if not state.waiting_for_c0_low:
            return result
            
        # Check for invalidation (stop loss hit)
        is_invalidated = (state.invalidation_level_low is not None and 
                         current_candle.close < state.invalidation_level_low)
        
        # Check for confirmation
        is_confirmed = (state.c1_close_for_low_confirm is not None and
                       current_candle.close > state.c1_close_for_low_confirm and
                       current_candle.is_bullish())
        
        if is_invalidated or is_confirmed:
            result['should_reset'] = True
            result['confirmed'] = is_confirmed
            result['invalidated'] = is_invalidated
            result['details'] = {
                'trigger_price': state.c1_close_for_low_confirm,
                'invalidation_level': state.invalidation_level_low,
                'current_close': current_candle.close,
                'current_candle_bullish': current_candle.is_bullish()
            }
            
        return result
        
    def check_c0_high_confirmation(self, current_candle: Candle, state: StateManager) -> Dict:
        """
        Exact implementation of Pine Script C0 high confirmation logic
        
        Returns:
            Dict with confirmation status, invalidation status, and details
        """
        result = {
            'confirmed': False,
            'invalidated': False,
            'should_reset': False,
            'details': {}
        }
        
        if not state.waiting_for_c0_high:
            return result
            
        # Check for invalidation (stop loss hit)
        is_invalidated = (state.invalidation_level_high is not None and 
                         current_candle.close > state.invalidation_level_high)
        
        # Check for confirmation
        is_confirmed = (state.c1_close_for_high_confirm is not None and
                       current_candle.close < state.c1_close_for_high_confirm and
                       current_candle.is_bearish())
        
        if is_invalidated or is_confirmed:
            result['should_reset'] = True
            result['confirmed'] = is_confirmed
            result['invalidated'] = is_invalidated
            result['details'] = {
                'trigger_price': state.c1_close_for_high_confirm,
                'invalidation_level': state.invalidation_level_high,
                'current_close': current_candle.close,
                'current_candle_bearish': current_candle.is_bearish()
            }
            
        return result

# ==================================================================================
# ROBUST MSS/BOS DETECTION ENGINE
# ==================================================================================

@dataclass
class StructureLevel:
    """Represents a significant market structure level"""
    price: float
    timestamp: int
    bar_index: int
    level_type: str  # "SWING_HIGH", "SWING_LOW", "MSS_HIGH", "MSS_LOW"
    strength: int    # 1-5 strength rating
    confirmed: bool = False
    broken: bool = False
    break_timestamp: Optional[int] = None

@dataclass
class MSSBOSEvent:
    """Represents an MSS or BOS event"""
    event_type: str  # "MSS_BULLISH", "MSS_BEARISH", "BOS_BULLISH", "BOS_BEARISH"
    broken_level: StructureLevel
    break_price: float
    break_timestamp: int
    break_bar_index: int
    confirmation_method: str  # "CLOSE_BREAK", "BODY_BREAK", "WICK_BREAK"
    strength: int  # 1-10 strength rating
    displacement: float  # How far price moved beyond the level
    speed: int  # How many candles to break the level

class MSSBOSDetectionEngine:
    """Robust Market Structure Shift and Break of Structure detection"""
    
    def __init__(self, config: PivotSystemConfig):
        self.config = config
        self.structure_levels: List[StructureLevel] = []
        self.mss_events: List[MSSBOSEvent] = []
        self.bos_events: List[MSSBOSEvent] = []
        self.current_trend = "NEUTRAL"  # "BULLISH", "BEARISH", "NEUTRAL"
        
    def detect_mss_bos(self, new_pivot: PivotData, pivot_history: List[PivotData], 
                      state: StateManager, current_candle: Candle) -> Dict:
        """
        Comprehensive MSS/BOS detection with structure analysis
        
        Returns:
            Dict with complete MSS/BOS analysis
        """
        result = {
            'mss_detected': False,
            'bos_detected': False,
            'mss_event': None,
            'bos_event': None,
            'structure_update': None,
            'trend_change': False,
            'new_trend': self.current_trend,
            'broken_levels': [],
            'details': {}
        }
        
        # Update structure levels with new pivot
        self._update_structure_levels(new_pivot, pivot_history)
        
        # Check for level breaks with current price action
        broken_levels = self._check_level_breaks(current_candle)
        
        if broken_levels:
            result['broken_levels'] = broken_levels
            
            # Analyze each broken level for MSS/BOS classification
            for level in broken_levels:
                mss_event, bos_event = self._classify_structure_break(level, current_candle, pivot_history)
                
                if mss_event:
                    result['mss_detected'] = True
                    result['mss_event'] = mss_event
                    self.mss_events.append(mss_event)
                    
                    # Check for trend change
                    old_trend = self.current_trend
                    self.current_trend = self._determine_new_trend(mss_event)
                    if old_trend != self.current_trend:
                        result['trend_change'] = True
                        result['new_trend'] = self.current_trend
                        
                if bos_event:
                    result['bos_detected'] = True
                    result['bos_event'] = bos_event
                    self.bos_events.append(bos_event)
        
        # Update market structure context
        result['details'] = {
            'current_trend': self.current_trend,
            'active_structure_levels': len([l for l in self.structure_levels if not l.broken]),
            'recent_mss_count': len([e for e in self.mss_events[-5:]]),
            'recent_bos_count': len([e for e in self.bos_events[-5:]]),
            'strongest_support': self._get_strongest_level("SUPPORT"),
            'strongest_resistance': self._get_strongest_level("RESISTANCE")
        }
        
        return result
    
    def _update_structure_levels(self, new_pivot: PivotData, pivot_history: List[PivotData]):
        """Update structure levels based on new pivot"""
        if not new_pivot.confirmed:
            return
            
        # Determine level strength based on pivot context
        strength = self._calculate_level_strength(new_pivot, pivot_history)
        
        # Create new structure level
        level_type = "SWING_HIGH" if new_pivot.pivot_type == PivotType.HIGH else "SWING_LOW"
        
        new_level = StructureLevel(
            price=new_pivot.price_level,
            timestamp=new_pivot.timestamp,
            bar_index=new_pivot.bar_index,
            level_type=level_type,
            strength=strength,
            confirmed=True
        )
        
        self.structure_levels.append(new_level)
        
        # Clean up old/weak structure levels
        self._cleanup_structure_levels()
    
    def _calculate_level_strength(self, pivot: PivotData, pivot_history: List[PivotData]) -> int:
        """Calculate strength of structure level (1-5)"""
        strength = 1
        
        # Base strength from pivot label
        if pivot.label in [PivotLabel.HH, PivotLabel.LL]:
            strength += 2  # Breaking previous extremes
        elif pivot.label in [PivotLabel.LH, PivotLabel.HL]:
            strength += 1  # Continuation patterns
            
        # Strength from time at level
        same_type_pivots = [p for p in pivot_history[-10:] 
                           if p.pivot_type == pivot.pivot_type and p.confirmed]
        if len(same_type_pivots) >= 2:
            # Check for multiple tests of similar levels
            price_tolerance = pivot.price_level * 0.001  # 0.1% tolerance
            nearby_pivots = [p for p in same_type_pivots 
                           if abs(p.price_level - pivot.price_level) <= price_tolerance]
            if len(nearby_pivots) >= 2:
                strength += 1  # Multiple tests = stronger level
                
        # Strength from distance from other levels
        if pivot_history:
            recent_same_type = [p for p in pivot_history[-5:] 
                              if p.pivot_type == pivot.pivot_type and p.confirmed]
            if recent_same_type:
                last_pivot = recent_same_type[-1]
                price_move_pct = abs(pivot.price_level - last_pivot.price_level) / last_pivot.price_level
                if price_move_pct > 0.02:  # > 2% move
                    strength += 1
                    
        return min(strength, 5)  # Cap at 5
    
    def _check_level_breaks(self, current_candle: Candle) -> List[StructureLevel]:
        """Check if current candle breaks any structure levels"""
        broken_levels = []
        
        for level in self.structure_levels:
            if level.broken:
                continue
                
            is_broken = False
            confirmation_method = ""
            
            if level.level_type in ["SWING_HIGH", "MSS_HIGH"]:
                # Check bullish break (break above resistance)
                if current_candle.close > level.price:
                    is_broken = True
                    confirmation_method = "CLOSE_BREAK"
                elif current_candle.high > level.price and current_candle.open < level.price:
                    is_broken = True
                    confirmation_method = "WICK_BREAK"
                    
            elif level.level_type in ["SWING_LOW", "MSS_LOW"]:
                # Check bearish break (break below support)
                if current_candle.close < level.price:
                    is_broken = True
                    confirmation_method = "CLOSE_BREAK"
                elif current_candle.low < level.price and current_candle.open > level.price:
                    is_broken = True
                    confirmation_method = "WICK_BREAK"
            
            if is_broken:
                level.broken = True
                level.break_timestamp = current_candle.timestamp
                broken_levels.append(level)
                
        return broken_levels
    
    def _classify_structure_break(self, broken_level: StructureLevel, current_candle: Candle, 
                                pivot_history: List[PivotData]) -> Tuple[Optional[MSSBOSEvent], Optional[MSSBOSEvent]]:
        """Classify structure break as MSS or BOS"""
        mss_event = None
        bos_event = None
        
        # Calculate displacement and speed
        displacement = abs(current_candle.close - broken_level.price)
        speed = 1  # Would need more context to calculate actual speed
        
        # Determine break direction
        is_bullish_break = current_candle.close > broken_level.price
        
        # MSS Classification Rules
        is_mss = self._is_market_structure_shift(broken_level, current_candle, pivot_history)
        
        if is_mss:
            event_type = "MSS_BULLISH" if is_bullish_break else "MSS_BEARISH"
            strength = min(broken_level.strength + 3, 10)  # MSS gets higher strength
            
            mss_event = MSSBOSEvent(
                event_type=event_type,
                broken_level=broken_level,
                break_price=current_candle.close,
                break_timestamp=current_candle.timestamp,
                break_bar_index=current_candle.bar_index,
                confirmation_method="CLOSE_BREAK" if abs(current_candle.close - broken_level.price) > 0 else "WICK_BREAK",
                strength=strength,
                displacement=displacement,
                speed=speed
            )
        else:
            # BOS Classification
            event_type = "BOS_BULLISH" if is_bullish_break else "BOS_BEARISH"
            strength = min(broken_level.strength + 1, 8)  # BOS gets moderate strength
            
            bos_event = MSSBOSEvent(
                event_type=event_type,
                broken_level=broken_level,
                break_price=current_candle.close,
                break_timestamp=current_candle.timestamp,
                break_bar_index=current_candle.bar_index,
                confirmation_method="CLOSE_BREAK" if abs(current_candle.close - broken_level.price) > 0 else "WICK_BREAK",
                strength=strength,
                displacement=displacement,
                speed=speed
            )
            
        return mss_event, bos_event
    
    def _is_market_structure_shift(self, broken_level: StructureLevel, current_candle: Candle, 
                                 pivot_history: List[PivotData]) -> bool:
        """Determine if break qualifies as MSS vs BOS"""
        # MSS Criteria:
        # 1. Breaking significant swing levels (strength >= 3)
        # 2. Breaking levels that haven't been broken in recent history
        # 3. Strong displacement beyond the level
        # 4. Changing the overall market structure pattern
        
        # Strength criterion
        if broken_level.strength < 3:
            return False
            
        # Check if this break changes market structure pattern
        recent_pivots = [p for p in pivot_history[-6:] if p.confirmed]
        if len(recent_pivots) < 4:
            return True  # Not enough data, assume MSS
            
        # Analyze recent pivot pattern
        highs = [p for p in recent_pivots if p.pivot_type == PivotType.HIGH]
        lows = [p for p in recent_pivots if p.pivot_type == PivotType.LOW]
        
        if len(highs) < 2 or len(lows) < 2:
            return True
            
        # Check for trend change
        is_bullish_break = current_candle.close > broken_level.price
        
        if is_bullish_break and broken_level.level_type in ["SWING_HIGH", "MSS_HIGH"]:
            # Breaking resistance - check if this creates higher high pattern
            last_two_highs = sorted(highs[-2:], key=lambda x: x.bar_index)
            if len(last_two_highs) == 2:
                if current_candle.close > max(h.price_level for h in last_two_highs):
                    return True  # Creating new higher high = MSS
                    
        elif not is_bullish_break and broken_level.level_type in ["SWING_LOW", "MSS_LOW"]:
            # Breaking support - check if this creates lower low pattern
            last_two_lows = sorted(lows[-2:], key=lambda x: x.bar_index)
            if len(last_two_lows) == 2:
                if current_candle.close < min(l.price_level for l in last_two_lows):
                    return True  # Creating new lower low = MSS
        
        return False  # Default to BOS
    
    def _determine_new_trend(self, mss_event: MSSBOSEvent) -> str:
        """Determine new trend after MSS"""
        if mss_event.event_type == "MSS_BULLISH":
            return "BULLISH"
        elif mss_event.event_type == "MSS_BEARISH":
            return "BEARISH"
        else:
            return self.current_trend
    
    def _get_strongest_level(self, level_type: str) -> Optional[Dict]:
        """Get strongest support or resistance level"""
        active_levels = [l for l in self.structure_levels if not l.broken]
        
        if level_type == "SUPPORT":
            support_levels = [l for l in active_levels if l.level_type in ["SWING_LOW", "MSS_LOW"]]
            if support_levels:
                strongest = max(support_levels, key=lambda x: x.strength)
                return {"price": strongest.price, "strength": strongest.strength}
        elif level_type == "RESISTANCE":
            resistance_levels = [l for l in active_levels if l.level_type in ["SWING_HIGH", "MSS_HIGH"]]
            if resistance_levels:
                strongest = max(resistance_levels, key=lambda x: x.strength)
                return {"price": strongest.price, "strength": strongest.strength}
                
        return None
    
    def _cleanup_structure_levels(self):
        """Clean up old and weak structure levels"""
        # Keep only the most recent 20 levels
        if len(self.structure_levels) > 20:
            # Sort by timestamp and keep most recent
            self.structure_levels.sort(key=lambda x: x.timestamp)
            self.structure_levels = self.structure_levels[-20:]
            
        # Keep only the most recent 10 MSS/BOS events
        if len(self.mss_events) > 10:
            self.mss_events = self.mss_events[-10:]
        if len(self.bos_events) > 10:
            self.bos_events = self.bos_events[-10:]

# ==================================================================================
# MAIN SYSTEM ORCHESTRATOR
# ==================================================================================

class AnasePivotSystem:
    """Main system that orchestrates all components - exact Pine Script behavior"""
    
    def __init__(self, config: PivotSystemConfig):
        self.config = config
        self.config.validate()
        
        self.state = StateManager()
        
        # Initialize engines
        self.pivot_engine = PivotDetectionEngine(config)
        self.ref_finder = ReferenceCandleFinder(config)
        self.c1_engine = C1DetectionEngine(config)
        self.c0_engine = C0ConfirmationEngine(config)
        self.mss_bos_engine = MSSBOSDetectionEngine(config)
        
        # Data storage
        self.candle_data: List[Candle] = []
        self.pivot_history: List[PivotData] = []
        self.c1_signals: List[C1Signal] = []
        self.confirmed_pivots: List[PivotData] = []
        
        # Performance tracking
        self.stats = {
            'total_candles_processed': 0,
            'c1_signals_generated': 0,
            'pivots_confirmed': 0,
            'mss_bos_detected': 0,
            'c1_invalidations': 0,
            'mss_events': 0,
            'bos_events': 0
        }
        
        logger.info(f"AnasePivotSystem initialized with config: {config}")
        
    def add_candle(self, timestamp: int, open_price: float, high: float, 
                  low: float, close: float) -> Dict:
        """
        Add new candle and process through complete pipeline
        
        Returns:
            Dict with all detection results for this candle
        """
        try:
            # Create candle with auto-incrementing bar_index
            bar_index = len(self.candle_data)
            candle = Candle(timestamp, open_price, high, low, close, bar_index)
            
            return self.process_candle(candle)
            
        except Exception as e:
            logger.error(f"Error adding candle: {e}")
            return {'error': str(e)}
        
    def process_candle(self, candle: Candle) -> Dict:
        """
        Main processing pipeline for each candle - exact Pine Script sequence:
        
        1. Update candle data storage
        2. Run pivot detection (find potential pivots)
        3. Find reference candles (for C1 detection)
        4. Run C1 detection (if not waiting for C0)
        5. Run C0 confirmation (if waiting for C0)
        6. Update state machine
        7. Run MSS/BOS detection (on confirmed pivots)
        8. Clean up old data (maintain max_bars_back)
        
        Returns:
            Complete results dictionary
        """
        results = {
            'candle': candle,
            'c1_signal': None,
            'c0_confirmation': None,
            'new_pivot': None,
            'mss_bos': None,
            'state_update': {},
            'stats': self.stats.copy()
        }
        
        try:
            # 1. Update candle data storage
            self.candle_data.append(candle)
            self.stats['total_candles_processed'] += 1
            
            # 2. Run pivot detection for potential pivots
            pivot_low_data = self._detect_potential_pivot(candle, False)  # Low pivot
            pivot_high_data = self._detect_potential_pivot(candle, True)   # High pivot
            
            # 3. Find reference candles for C1 detection
            ref_candle_low = self._find_reference_candle(False)   # For bullish C1
            ref_candle_high = self._find_reference_candle(True)   # For bearish C1
            
            # 4. Run C1 detection (if not waiting for C0)
            if not self.state.waiting_for_c0_low and not self.state.waiting_for_c0_high:
                c1_signal = self._run_c1_detection(candle, ref_candle_low, ref_candle_high, 
                                                 pivot_low_data, pivot_high_data)
                if c1_signal:
                    results['c1_signal'] = c1_signal
                    self.c1_signals.append(c1_signal)
                    self.stats['c1_signals_generated'] += 1
                    
            # 5. Run C0 confirmation (if waiting for C0)
            c0_result = self._run_c0_confirmation(candle)
            if c0_result['should_reset']:
                results['c0_confirmation'] = c0_result
                
                # If confirmed, finalize the pivot
                if c0_result['confirmed']:
                    new_pivot = self._finalize_pivot(candle)
                    if new_pivot:
                        results['new_pivot'] = new_pivot
                        self.stats['pivots_confirmed'] += 1
                        
                        # 7. Run MSS/BOS detection on new confirmed pivot
                        if self.config.enable_mss_bos_detection:
                            mss_bos_result = self.mss_bos_engine.detect_mss_bos(
                                new_pivot, self.confirmed_pivots, self.state, candle)
                            if mss_bos_result['mss_detected'] or mss_bos_result['bos_detected']:
                                results['mss_bos'] = mss_bos_result
                                self.stats['mss_bos_detected'] += 1
                                if mss_bos_result['mss_detected']:
                                    self.stats['mss_events'] += 1
                                if mss_bos_result['bos_detected']:
                                    self.stats['bos_events'] += 1
                else:
                    self.stats['c1_invalidations'] += 1
                    
                # Reset state after confirmation or invalidation
                self.state.reset_c1_state()
                
            # 6. Update state tracking
            results['state_update'] = {
                'waiting_for_c0_low': self.state.waiting_for_c0_low,
                'waiting_for_c0_high': self.state.waiting_for_c0_high,
                'looking_for': self.state.looking_for.value,
                'last_pivot_bar': self.state.last_pivot_bar
            }
            
            # 8. Clean up old data
            self._cleanup_old_data()
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing candle: {e}")
            results['error'] = str(e)
            return results
            
    def _detect_potential_pivot(self, candle: Candle, is_high: bool) -> Optional[Dict]:
        """Detect potential pivot at current or recent bars"""
        # Calculate appropriate lookback and offset for current state
        if is_high and self.state.waiting_for_c0_high:
            lookback = (self.config.pivot_lookback if self.state.last_pivot_bar is None 
                       else candle.bar_index - self.state.c1_bar_index_for_high_confirm)
            offset = candle.bar_index - self.state.c1_bar_index_for_high_confirm
        elif not is_high and self.state.waiting_for_c0_low:
            lookback = (self.config.pivot_lookback if self.state.last_pivot_bar is None
                       else candle.bar_index - self.state.c1_bar_index_for_low_confirm)
            offset = candle.bar_index - self.state.c1_bar_index_for_low_confirm
        else:
            lookback = self.config.pivot_lookback
            offset = 1
            
        # Ensure positive lookback
        lookback = max(1, lookback)
        offset = max(0, offset)
            
        # Find potential pivot
        pivot_bar, pivot_price = self.pivot_engine.find_pivot_candle_by_close(
            self.candle_data, offset, lookback, is_high)
            
        if pivot_bar is not None and pivot_price is not None:
            return {
                'bar_index': pivot_bar,
                'price': pivot_price,
                'type': PivotType.HIGH if is_high else PivotType.LOW
            }
            
        return None
        
    def _find_reference_candle(self, is_for_high: bool) -> Optional[ReferenceCandle]:
        """Find reference candle for C1 detection"""
        if len(self.candle_data) < 2:
            return None
            
        # Calculate safe lookback
        safe_lookback = self.ref_finder.get_safe_lookback(
            self.config.pivot_lookback, self.state.last_pivot_bar, 
            self.candle_data[-1].bar_index)
            
        # Use configured method to find reference
        if self.config.reference_method == ReferenceMethod.REVERSAL_OPEN:
            return self.ref_finder.find_ref_by_reversal_open(
                self.candle_data, 1, safe_lookback, is_for_high)
        else:
            return self.ref_finder.find_ref_by_wick(
                self.candle_data, 1, safe_lookback, is_for_high)
                
    def _run_c1_detection(self, candle: Candle, ref_low: Optional[ReferenceCandle], 
                         ref_high: Optional[ReferenceCandle], pivot_low_data: Optional[Dict],
                         pivot_high_data: Optional[Dict]) -> Optional[C1Signal]:
        """Run C1 detection for both directions"""
        
        # Check for bullish C1 (if looking for lows or any)
        if (self.state.looking_for in [LookingFor.LOW, LookingFor.ANY] and 
            pivot_low_data and ref_low):
            
            if self.c1_engine.detect_bullish_c1(candle, ref_low, None, self.state):
                # Update state for C0 confirmation
                self.state.waiting_for_c0_low = True
                self.state.c1_close_for_low_confirm = candle.close
                self.state.c1_bar_index_for_low_confirm = candle.bar_index
                self.state.invalidation_level_low = candle.open
                
                return C1Signal(
                    detected=True,
                    signal_type=SignalType.BULLISH_C1,
                    trigger_price=candle.close,
                    invalidation_level=candle.open,
                    reference_candle=ref_low,
                    timestamp=candle.timestamp,
                    bar_index=candle.bar_index
                )
                
        # Check for bearish C1 (if looking for highs or any)
        if (self.state.looking_for in [LookingFor.HIGH, LookingFor.ANY] and 
            pivot_high_data and ref_high):
            
            if self.c1_engine.detect_bearish_c1(candle, ref_high, None, self.state):
                # Update state for C0 confirmation
                self.state.waiting_for_c0_high = True
                self.state.c1_close_for_high_confirm = candle.close
                self.state.c1_bar_index_for_high_confirm = candle.bar_index
                self.state.invalidation_level_high = candle.open
                
                return C1Signal(
                    detected=True,
                    signal_type=SignalType.BEARISH_C1,
                    trigger_price=candle.close,
                    invalidation_level=candle.open,
                    reference_candle=ref_high,
                    timestamp=candle.timestamp,
                    bar_index=candle.bar_index
                )
                
        return None
        
    def _run_c0_confirmation(self, candle: Candle) -> Dict:
        """Run C0 confirmation checks"""
        # Check for low confirmation
        if self.state.waiting_for_c0_low:
            return self.c0_engine.check_c0_low_confirmation(candle, self.state)
            
        # Check for high confirmation
        if self.state.waiting_for_c0_high:
            return self.c0_engine.check_c0_high_confirmation(candle, self.state)
            
        return {'should_reset': False}
        
    def _finalize_pivot(self, candle: Candle) -> Optional[PivotData]:
        """Finalize and confirm a pivot"""
        if self.state.waiting_for_c0_low and self.state.c1_bar_index_for_low_confirm:
            # Find the pivot bar for the low
            lookback = (self.config.pivot_lookback if self.state.last_pivot_bar is None
                       else candle.bar_index - self.state.c1_bar_index_for_low_confirm)
            offset = max(0, candle.bar_index - self.state.c1_bar_index_for_low_confirm)
            lookback = max(1, lookback)
            
            pivot_bar_index, pivot_price = self.pivot_engine.find_pivot_candle_by_close(
                self.candle_data, offset, lookback, False)
                
            if pivot_bar_index and pivot_price:
                # Validate and label the pivot
                is_valid, label, validated_price = self.pivot_engine.validate_and_label_pivot(
                    self.candle_data, pivot_bar_index, False, self.confirmed_pivots)
                    
                if is_valid:
                    pivot = PivotData(
                        bar_index=pivot_bar_index,
                        timestamp=candle.timestamp,  # Use current timestamp for confirmation
                        price_level=validated_price,
                        pivot_type=PivotType.LOW,
                        label=label,
                        confirmed=True
                    )
                    
                    self.confirmed_pivots.append(pivot)
                    self.state.last_pivot_bar = pivot_bar_index
                    self.state.last_confirmed_low_price = validated_price
                    self.state.looking_for = LookingFor.HIGH
                    
                    return pivot
                    
        elif self.state.waiting_for_c0_high and self.state.c1_bar_index_for_high_confirm:
            # Find the pivot bar for the high
            lookback = (self.config.pivot_lookback if self.state.last_pivot_bar is None
                       else candle.bar_index - self.state.c1_bar_index_for_high_confirm)
            offset = max(0, candle.bar_index - self.state.c1_bar_index_for_high_confirm)
            lookback = max(1, lookback)
            
            pivot_bar_index, pivot_price = self.pivot_engine.find_pivot_candle_by_close(
                self.candle_data, offset, lookback, True)
                
            if pivot_bar_index and pivot_price:
                # Validate and label the pivot
                is_valid, label, validated_price = self.pivot_engine.validate_and_label_pivot(
                    self.candle_data, pivot_bar_index, True, self.confirmed_pivots)
                    
                if is_valid:
                    pivot = PivotData(
                        bar_index=pivot_bar_index,
                        timestamp=candle.timestamp,  # Use current timestamp for confirmation
                        price_level=validated_price,
                        pivot_type=PivotType.HIGH,
                        label=label,
                        confirmed=True
                    )
                    
                    self.confirmed_pivots.append(pivot)
                    self.state.last_pivot_bar = pivot_bar_index
                    self.state.last_confirmed_high_price = validated_price
                    self.state.looking_for = LookingFor.LOW
                    
                    return pivot
                    
        return None
        
    def _cleanup_old_data(self):
        """Clean up old data to maintain performance"""
        max_candles = self.config.max_bars_back
        
        if len(self.candle_data) > max_candles:
            # Keep only the most recent candles
            self.candle_data = self.candle_data[-max_candles:]
            
        # Clean up old pivots
        if len(self.confirmed_pivots) > self.config.max_pivots_stored:
            self.confirmed_pivots = self.confirmed_pivots[-self.config.max_pivots_stored:]
            
        # Clean up old C1 signals
        if len(self.c1_signals) > 100:  # Keep last 100 signals
            self.c1_signals = self.c1_signals[-100:]
            
    def get_current_state(self) -> Dict:
        """Get current system state for debugging/monitoring"""
        return {
            'state': {
                'waiting_for_c0_low': self.state.waiting_for_c0_low,
                'waiting_for_c0_high': self.state.waiting_for_c0_high,
                'looking_for': self.state.looking_for.value,
                'last_pivot_bar': self.state.last_pivot_bar,
                'c1_close_for_low_confirm': self.state.c1_close_for_low_confirm,
                'c1_close_for_high_confirm': self.state.c1_close_for_high_confirm,
                'current_trend': getattr(self.mss_bos_engine, 'current_trend', 'NEUTRAL')
            },
            'data_counts': {
                'candles': len(self.candle_data),
                'confirmed_pivots': len(self.confirmed_pivots),
                'c1_signals': len(self.c1_signals),
                'structure_levels': len(getattr(self.mss_bos_engine, 'structure_levels', [])),
                'mss_events': len(getattr(self.mss_bos_engine, 'mss_events', [])),
                'bos_events': len(getattr(self.mss_bos_engine, 'bos_events', []))
            },
            'last_candle': self.candle_data[-1] if self.candle_data else None,
            'last_confirmed_pivot': self.confirmed_pivots[-1] if self.confirmed_pivots else None,
            'last_c1_signal': self.c1_signals[-1] if self.c1_signals else None
        }
        
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        total_candles = self.stats['total_candles_processed']
        
        return {
            **self.stats,
            'c1_detection_rate': (self.stats['c1_signals_generated'] / max(total_candles, 1)) * 100,
            'c1_success_rate': (self.stats['pivots_confirmed'] / max(self.stats['c1_signals_generated'], 1)) * 100,
            'invalidation_rate': (self.stats['c1_invalidations'] / max(self.stats['c1_signals_generated'], 1)) * 100,
            'mss_rate': (self.stats['mss_events'] / max(total_candles, 1)) * 100,
            'bos_rate': (self.stats['bos_events'] / max(total_candles, 1)) * 100
        }
    
    def get_active_structure_levels(self) -> List[Dict]:
        """Get current active structure levels"""
        if hasattr(self.mss_bos_engine, 'structure_levels'):
            active_levels = [l for l in self.mss_bos_engine.structure_levels if not l.broken]
            return [
                {
                    'price': level.price,
                    'type': level.level_type,
                    'strength': level.strength,
                    'timestamp': level.timestamp
                }
                for level in active_levels
            ]
        return []
    
    def get_recent_mss_bos_events(self, count: int = 5) -> Dict:
        """Get recent MSS/BOS events"""
        result = {'mss_events': [], 'bos_events': []}
        
        if hasattr(self.mss_bos_engine, 'mss_events'):
            recent_mss = self.mss_bos_engine.mss_events[-count:]
            result['mss_events'] = [
                {
                    'type': event.event_type,
                    'price': event.break_price,
                    'timestamp': event.break_timestamp,
                    'strength': event.strength,
                    'displacement': event.displacement
                }
                for event in recent_mss
            ]
            
        if hasattr(self.mss_bos_engine, 'bos_events'):
            recent_bos = self.mss_bos_engine.bos_events[-count:]
            result['bos_events'] = [
                {
                    'type': event.event_type,
                    'price': event.break_price,
                    'timestamp': event.break_timestamp,
                    'strength': event.strength,
                    'displacement': event.displacement
                }
                for event in recent_bos
            ]
            
        return result


