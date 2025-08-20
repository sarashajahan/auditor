import asyncio
import json
import sqlite3
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass
from enum import Enum
import hashlib
import aiohttp
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # default fallback model

openai_client = OpenAI(api_key=api_key)

SQLITE_DB_PATH = "sox_agent.db"


# ---------- Async SQLite Helpers ----------

# Run DB operation in thread pool to avoid blocking event loop
async def run_db_query(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


# ---------- Initialize SQLite DB and tables ----------

def init_sqlite_db_sync():
    """Synchronous DB init (called via async wrapper below)"""
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS vendors (
                vendor_name TEXT PRIMARY KEY,
                status TEXT,
                allowed_categories TEXT,
                regions TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS exceptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT,
                line_item_description TEXT,
                reason TEXT,
                active INTEGER
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS approval_thresholds (
                category TEXT PRIMARY KEY,
                max_approval_amount REAL
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT,
                user_id TEXT,
                action TEXT,
                details TEXT,
                timestamp TEXT,
                hash TEXT UNIQUE
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS exchange_rates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_currency TEXT,
                to_currency TEXT,
                rate REAL,
                date TEXT,
                UNIQUE(from_currency, to_currency, date)
            )
        ''')
        # Seed default approval thresholds if table is empty
        c.execute('SELECT COUNT(*) FROM approval_thresholds')
        cnt = c.fetchone()[0]
        if not cnt:
            defaults = {
                'travel': 500.0,
                'meals': 100.0,
                'accommodation': 1000.0,
                'office_supplies': 300.0,
                'unknown': 0.0,
            }
            for cat, amt in defaults.items():
                c.execute(
                    'INSERT OR REPLACE INTO approval_thresholds (category, max_approval_amount) VALUES (?, ?)',
                    (cat, amt)
                )
        conn.commit()

async def init_sqlite_db():
    await run_db_query(init_sqlite_db_sync)


# ---------- Helper Functions ----------

def decimal_round(value: Decimal, places=2) -> Decimal:
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    return value.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)

def hash_record(record_str: str) -> str:
    return hashlib.sha256(record_str.encode('utf-8')).hexdigest()


# --- Vendor Policy Check ---

def _normalize_text(text: str) -> str:
    return text.strip().lower() if isinstance(text, str) else ""

def _normalize_list(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [_normalize_text(v) for v in raw if _normalize_text(v)]
    # assume comma-separated string
    return [_normalize_text(v) for v in str(raw).split(",") if _normalize_text(v)]

def check_vendor_policy_sync(vendor_name, category, jurisdiction):
    norm_vendor = _normalize_text(vendor_name)
    norm_category = _normalize_text(category)
    norm_jurisdiction = _normalize_text(jurisdiction)

    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        c = conn.cursor()
        # Case-insensitive match on vendor_name
        c.execute(
            "SELECT status, allowed_categories, regions FROM vendors WHERE lower(vendor_name)=?",
            (norm_vendor,)
        )
        row = c.fetchone()
    if not row:
        return None, f"Vendor '{vendor_name}' unknown, manual review recommended."

    status, allowed_cat_raw, allowed_regions_raw = row
    if _normalize_text(status) == "blacklisted":
        return False, f"Vendor '{vendor_name}' is blacklisted."

    # Parse lists (JSON or CSV) and normalize entries
    try:
        allowed_categories = _normalize_list(json.loads(allowed_cat_raw))
    except Exception:
        allowed_categories = _normalize_list(allowed_cat_raw)
    try:
        allowed_regions = _normalize_list(json.loads(allowed_regions_raw))
    except Exception:
        allowed_regions = _normalize_list(allowed_regions_raw)

    if norm_category not in allowed_categories:
        return False, f"Vendor '{vendor_name}' not authorized for category '{category}'."
    if norm_jurisdiction not in allowed_regions:
        return False, f"Vendor '{vendor_name}' not authorized in region '{jurisdiction}'."
    return True, "Vendor compliant."

def add_or_update_vendor_sync(vendor_name: str, status: str = "approved", allowed_categories=None, regions=None):
    """Upsert a vendor record into the vendors table.
    - allowed_categories: list[str] or comma string; stored as JSON
    - regions: list[str] or comma string; stored as JSON
    """
    allowed_categories = _normalize_list(allowed_categories)
    regions = _normalize_list(regions)
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO vendors (vendor_name, status, allowed_categories, regions)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(vendor_name) DO UPDATE SET
                status=excluded.status,
                allowed_categories=excluded.allowed_categories,
                regions=excluded.regions
            """,
            (
                vendor_name.strip(),
                status.strip(),
                json.dumps(allowed_categories),
                json.dumps(regions),
            ),
        )
        conn.commit()

async def add_or_update_vendor(vendor_name: str, status: str = "approved", allowed_categories=None, regions=None):
    return await run_db_query(add_or_update_vendor_sync, vendor_name, status, allowed_categories, regions)

async def check_vendor_policy(vendor_name, category, jurisdiction):
    return await run_db_query(check_vendor_policy_sync, vendor_name, category, jurisdiction)


# --- Exception Check ---

def check_exceptions_sync(report_id, line_item_description):
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""SELECT reason FROM exceptions WHERE report_id=? AND line_item_description=? AND active=1""",
                  (report_id, line_item_description))
        row = c.fetchone()
    if row:
        return True, row[0] or "Exception granted"
    return False, ""

async def check_exceptions(report_id, line_item_description):
    return await run_db_query(check_exceptions_sync, report_id, line_item_description)


# --- Approval Threshold Fetch ---

def get_approval_threshold_sync(category):
    """Fetch approval threshold by category with case-insensitive lookup and 'unknown' fallback."""
    cat_norm = str(category).strip().lower()
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        c = conn.cursor()
        # Case-insensitive match on provided category
        c.execute("SELECT max_approval_amount FROM approval_thresholds WHERE lower(category)=?", (cat_norm,))
        row = c.fetchone()
        if not row or row[0] is None:
            # Fallback to 'unknown' category if specific one is missing
            c.execute("SELECT max_approval_amount FROM approval_thresholds WHERE lower(category)='unknown'")
            row = c.fetchone()
    if row and row[0] is not None:
        return Decimal(str(row[0]))
    return Decimal("0")


async def get_approval_threshold(category):
    return await run_db_query(get_approval_threshold_sync, category)

def set_approval_threshold_sync(category: str, max_amount):
    """Upsert an approval threshold value for a category (case-insensitive)."""
    cat_norm = str(category).strip().lower()
    amount = float(Decimal(str(max_amount)))
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO approval_thresholds (category, max_approval_amount) VALUES (?, ?)",
            (cat_norm, amount),
        )
        conn.commit()

async def set_approval_threshold(category: str, max_amount):
    return await run_db_query(set_approval_threshold_sync, category, max_amount)


# --- Expense Dates Validation (sync) ---

def validate_expense_dates(expense_date: datetime, submission_date: datetime):
    today = datetime.utcnow().date()
    if expense_date.date() > today:
        return False, "Expense date cannot be in the future."
    if (today - expense_date.date()).days > 365:
        return False, "Expense date is too old."
    if (submission_date.date() - expense_date.date()).days > 60:
        return False, "Expense submitted too late."
    return True, "Dates valid."


# ---------- Enums and Data Classes (unchanged) ----------

class Role(Enum):
    EMPLOYEE = "employee"
    MANAGER = "manager"
    DIRECTOR = "director"
    CFO = "cfo"
    AUDITOR = "auditor"

class ExpenseCategory(Enum):
    TRAVEL = "travel"
    MEALS = "meals"
    ACCOMMODATION = "accommodation"
    OFFICE_SUPPLIES = "office_supplies"
    UNKNOWN = "unknown"

class ApprovalStatus(Enum):
    PENDING = "pending"
    MANAGER_APPROVED = "manager_approved"
    DIRECTOR_APPROVED = "director_approved"
    CFO_APPROVED = "cfo_approved"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"

@dataclass
class User:
    user_id: str
    role: Role

@dataclass
class LineItem:
    description: str
    quantity: Decimal
    unit_price: Decimal
    tax_rate: Decimal
    category: ExpenseCategory
    currency: str = "USD"
    jurisdiction: str = "US"
    vendor: str = None

    @property
    def subtotal(self):
        return decimal_round(self.quantity * self.unit_price)

    @property
    def tax_amount(self):
        return decimal_round(self.subtotal * self.tax_rate)

    @property
    def total(self):
        return decimal_round(self.subtotal + self.tax_amount)

@dataclass
class ExpenseReport:
    report_id: str
    employee: User
    submission_date: datetime
    expense_date: datetime
    line_items: list
    currency: str = "USD"
    jurisdiction: str = "US"
    claimed_total: Decimal = None
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    risk_score: float = 0.0
    invoice_metadata: dict = None

@dataclass
class ValidationResult:
    is_valid: bool
    violations: list
    warnings: list
    risk_score: float
    recommended_action: str


# ---------- Access Control (unchanged) ----------

class AccessControl:
    role_hierarchy = {
        Role.EMPLOYEE: 1,
        Role.MANAGER: 2,
        Role.DIRECTOR: 3,
        Role.CFO: 4,
        Role.AUDITOR: 5,
    }

    def can_approve(self, user: User, amount: Decimal) -> bool:
        if user.role == Role.MANAGER and amount <= 500:
            return True
        elif user.role == Role.DIRECTOR and amount <= 5000:
            return True
        elif user.role == Role.CFO:
            return True
        return False

    def can_submit(self, user: User) -> bool:
        return user.role == Role.EMPLOYEE

    def can_audit(self, user: User) -> bool:
        return user.role == Role.AUDITOR


# ---------- Audit Trail ----------

def log_audit_sync(report_id: str, user_id: str, action: str, details: dict):
    timestamp = datetime.utcnow().isoformat()
    details_str = json.dumps(details, sort_keys=True)
    record_str = f"{report_id}{user_id}{action}{details_str}{timestamp}"
    record_hash = hash_record(record_str)
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        c = conn.cursor()
        try:
            c.execute('''INSERT INTO audit_logs (report_id, user_id, action, details, timestamp, hash)
                         VALUES (?, ?, ?, ?, ?, ?)''',
                      (report_id, user_id, action, details_str, timestamp, record_hash))
            conn.commit()
        except sqlite3.IntegrityError:
            pass

async def log_audit(report_id: str, user_id: str, action: str, details: dict):
    await run_db_query(log_audit_sync, report_id, user_id, action, details)


# ---------- GPT-Based Tax Validation Engine (replaces TaxRuleEngine) ----------

class GPTTaxValidationEngine:
    def __init__(self):
        self.openai_client = openai_client
        self.model_name = model_name

    async def identify_commodity_and_tax_rate(self, item_description: str, jurisdiction: str, amount: Decimal) -> dict:
        """
        Use GPT to identify the commodity type and determine the appropriate tax rate
        based on the item description, jurisdiction, and amount.
        """
        prompt = f"""
You are a tax compliance expert specializing in commodity classification and tax rate determination.

Given the following expense item, please:

1. Identify the commodity type (e.g., food, electronics, services, etc.)
2. Determine the appropriate tax rate based on the jurisdiction and commodity type
3. Validate if the provided tax rate is reasonable for this commodity and jurisdiction
4. Provide a confidence score for your assessment

Expense Description: "{item_description}"
Jurisdiction: {jurisdiction}
Amount: {amount}

Consider:
- Local tax laws and regulations for the jurisdiction
- Commodity-specific tax rates (e.g., food items often have different rates)
- Standard tax rates for the jurisdiction
- Any special tax categories that might apply

Output JSON with keys:
- commodity_type: string (the identified commodity category)
- recommended_tax_rate: decimal (the appropriate tax rate as a decimal, e.g., 0.08 for 8%)
- is_tax_rate_reasonable: boolean (whether the provided tax rate seems appropriate)
- confidence_score: float (0.0 to 1.0, how confident you are in your assessment)
- explanation: string (brief explanation of your reasoning)
- tax_category: string (e.g., "standard", "reduced", "zero-rated", "exempt")
"""

        try:
            # Check if OpenAI client is properly configured
            if not self.openai_client or not hasattr(self.openai_client, 'api_key') or not self.openai_client.api_key:
                raise Exception("OpenAI client not properly configured. Please check OPENAI_API_KEY environment variable.")
            
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(None, self._run_gpt_call_sync, prompt)
            
            # Parse the GPT response
            result = json.loads(content)
            
            # Ensure the tax rate is a valid decimal
            if 'recommended_tax_rate' in result:
                try:
                    result['recommended_tax_rate'] = Decimal(str(result['recommended_tax_rate']))
                except (ValueError, TypeError):
                    result['recommended_tax_rate'] = Decimal('0.0')
            
            return result
            
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors
            print(f"JSON parsing error in GPT response: {e}")
            print(f"Raw response: {content if 'content' in locals() else 'No content'}")
            return self._get_fallback_response(item_description, jurisdiction, f"JSON parsing failed: {str(e)}")
        except Exception as e:
            # Handle other errors
            print(f"GPT API call failed: {str(e)}")
            return self._get_fallback_response(item_description, jurisdiction, f"GPT validation failed: {str(e)}")

    def _run_gpt_call_sync(self, prompt: str):
        """Synchronous wrapper for GPT API call"""
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1  # Lower temperature for more consistent tax-related responses
        )
        return response.choices[0].message.content

    def _get_fallback_response(self, item_description: str, jurisdiction: str, error_msg: str) -> dict:
        """
        Generate a fallback response when GPT validation fails.
        Uses basic keyword matching to identify commodity types.
        """
        description_lower = item_description.lower()
        
        # Basic commodity identification based on keywords
        commodity_type = "unknown"
        confidence_score = 0.3  # Low confidence for fallback
        
        if any(word in description_lower for word in [
            # Sports gear and related terms
            'soccer', 'football', 'basketball', 'tennis', 'cricket', 'hockey', 'golf',
            'cleats', 'studs', 'spikes', 'jersey', 'kit', 'shin guard', 'shin-guard', 'shin pads',
            'sports', 'athletic', 'training gear', 'sportswear', 'gym equipment', 'yoga mat', 'dumbbell'
        ]):
            commodity_type = "sports_equipment"
            confidence_score = 0.75
        elif any(word in description_lower for word in [
            # Footwear
            'shoes', 'boots', 'footwear', 'sneakers', 'trainers', 'sandals', 'heels', 'loafers'
        ]):
            commodity_type = "footwear"
            confidence_score = 0.8
        elif any(word in description_lower for word in [
            # Meals & food
            'lunch', 'dinner', 'breakfast', 'meal', 'meals', 'restaurant', 'cafe', 'coffee', 'tea', 'snack',
            'food', 'beverage', 'beverages', 'catering', 'bar', 'pub', 'brewery', 'bistro', 'canteen', 'deli',
            'takeaway', 'take-away', 'take out', 'take-out', 'delivery', 'groceries', 'grocery'
        ]):
            commodity_type = "food"
            confidence_score = 0.85
        elif any(word in description_lower for word in [
            # Accommodation
            'hotel', 'motel', 'lodging', 'accommodation', 'inn', 'resort', 'hostel', 'guesthouse', 'guest house',
            'b&b', 'bnb', 'airbnb', 'stay', 'suite', 'room night', 'room-nights', 'night stay', 'night-stay'
        ]):
            commodity_type = "accommodation"
            confidence_score = 0.9
        elif any(word in description_lower for word in [
            # Transportation
            'uber', 'taxi', 'cab', 'ride', 'lyft', 'transport', 'transportation', 'tolls', 'parking',
            'train', 'rail', 'railway', 'bus', 'coach', 'metro', 'subway', 'tram', 'ferry',
            'flight', 'airfare', 'airline', 'ticket', 'boarding pass', 'boarding-pass', 'check-in',
            'car rental', 'vehicle rental', 'rent-a-car', 'hire car', 'mileage', 'fuel', 'gas', 'diesel', 'petrol'
        ]):
            commodity_type = "transportation"
            confidence_score = 0.9
        elif any(word in description_lower for word in [
            # Office supplies & stationery
            'office', 'supplies', 'stationery', 'paper', 'copy paper', 'a4 paper', 'pens', 'pen', 'pencils',
            'markers', 'highlighter', 'highlighters', 'notebook', 'notebooks', 'notepad', 'pads', 'sticky notes',
            'post-it', 'post its', 'stapler', 'staples', 'tape', 'glue', 'scissors', 'envelope', 'envelopes',
            'folder', 'folders', 'binder', 'binders', 'laminating', 'whiteboard', 'eraser', 'clips', 'push pins',
            'toner', 'ink', 'cartridge', 'cartridges', 'printer', 'printer paper', 'label', 'labels'
        ]):
            commodity_type = "office_supplies"
            confidence_score = 0.9
        
        # Basic tax rate based on jurisdiction and commodity
        recommended_tax_rate = self._get_basic_tax_rate(jurisdiction, commodity_type)
        
        return {
            "commodity_type": commodity_type,
            "recommended_tax_rate": recommended_tax_rate,
            "is_tax_rate_reasonable": True,
            "confidence_score": confidence_score,
            "explanation": f"Fallback identification used due to: {error_msg}. Identified as {commodity_type} based on keywords.",
            "tax_category": "standard"
        }

    def _get_basic_tax_rate(self, jurisdiction: str, commodity_type: str) -> Decimal:
        """
        Get basic tax rates for common jurisdictions and commodities.
        Normalizes jurisdictions like "US-NY" -> "US".
        This serves as a fallback when GPT is unavailable.
        """
        # Normalize jurisdiction (e.g., "US-NY" -> "US")
        base_jurisdiction = (jurisdiction or "").strip().upper().split("-")[0]

        basic_rates = {
            "US": {
                "sports_equipment": Decimal("0.08"),
                "footwear": Decimal("0.08"),
                "food": Decimal("0.08"),
                "accommodation": Decimal("0.12"),
                "transportation": Decimal("0.08"),
                "office_supplies": Decimal("0.06"),
                "unknown": Decimal("0.08")
            },
            "EU": {
                "sports_equipment": Decimal("0.20"),
                "footwear": Decimal("0.20"),
                "food": Decimal("0.15"),
                "accommodation": Decimal("0.18"),
                "transportation": Decimal("0.20"),
                "office_supplies": Decimal("0.07"),
                "unknown": Decimal("0.20")
            },
            "CA": {
                "sports_equipment": Decimal("0.13"),
                "footwear": Decimal("0.13"),
                "food": Decimal("0.05"),
                "accommodation": Decimal("0.13"),
                "transportation": Decimal("0.13"),
                "office_supplies": Decimal("0.13"),
                "unknown": Decimal("0.13")
            }
        }

        rates_for_region = basic_rates.get(base_jurisdiction)
        if not rates_for_region:
            # Unknown region: default to US baseline
            rates_for_region = basic_rates["US"]
        return rates_for_region.get(commodity_type, rates_for_region["unknown"])

    async def test_api_connectivity(self) -> dict:
        """
        Test the OpenAI API connectivity and return diagnostic information.
        """
        diagnostic = {
            "api_key_configured": False,
            "client_initialized": False,
            "model_name": self.model_name,
            "test_result": "unknown",
            "error_message": None
        }
        
        try:
            # Check if API key is configured
            if hasattr(self.openai_client, 'api_key') and self.openai_client.api_key:
                diagnostic["api_key_configured"] = True
                diagnostic["client_initialized"] = True
            else:
                diagnostic["error_message"] = "OpenAI API key not configured"
                return diagnostic
            
            # Test a simple API call
            test_prompt = "Respond with just the word 'test'"
            try:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, self._run_gpt_call_sync, test_prompt)
                if response and "test" in response.lower():
                    diagnostic["test_result"] = "success"
                else:
                    diagnostic["test_result"] = "unexpected_response"
                    diagnostic["error_message"] = f"Unexpected response: {response}"
            except Exception as e:
                diagnostic["test_result"] = "api_call_failed"
                diagnostic["error_message"] = str(e)
                
        except Exception as e:
            diagnostic["error_message"] = str(e)
        
        return diagnostic

    async def validate_tax_rate(self, item: LineItem, jurisdiction: str) -> dict:
        """
        Validate the tax rate for a specific line item using GPT analysis.
        Returns validation result with recommendations.
        """
        validation_result = await self.identify_commodity_and_tax_rate(
            item.description, 
            jurisdiction, 
            item.total
        )
        
        # Calculate tax rate deviation
        provided_rate = item.tax_rate
        recommended_rate = validation_result.get('recommended_tax_rate', Decimal('0.0'))
        
        if recommended_rate > 0:
            rate_deviation = abs(provided_rate - recommended_rate) / recommended_rate
        else:
            rate_deviation = 0.0
        
        # Determine if the tax rate is acceptable
        is_acceptable = (
            validation_result.get('is_tax_rate_reasonable', True) and
            rate_deviation <= 0.05  # Allow 5% deviation
        )
        
        return {
            **validation_result,
            "provided_tax_rate": provided_rate,
            "rate_deviation": rate_deviation,
            "is_acceptable": is_acceptable,
            "validation_status": "PASS" if is_acceptable else "FAIL"
        }

    async def get_commodity_tax_rate(self, commodity_type: str, jurisdiction: str) -> Decimal:
        """
        Get the standard tax rate for a specific commodity type in a jurisdiction.
        This can be used as a fallback or reference point.
        """
        prompt = f"""
You are a tax expert. What is the standard tax rate for {commodity_type} in {jurisdiction}?

Return only the decimal value (e.g., 0.08 for 8%) or 0.0 if unknown.
"""
        
        try:
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(None, self._run_gpt_call_sync, prompt)
            
            # Extract numeric value from response
            import re
            rate_match = re.search(r'0\.\d+', content)
            if rate_match:
                return Decimal(rate_match.group())
            else:
                return Decimal('0.0')
                
        except Exception:
            return Decimal('0.0')

    async def analyze_expense_report_commodities(self, report: ExpenseReport) -> dict:
        """
        Analyze all line items in an expense report to identify commodities and validate tax rates.
        Returns a comprehensive analysis summary.
        """
        analysis_results = []
        total_items = len(report.line_items)
        passed_validation = 0
        
        for item in report.line_items:
            validation_result = await self.validate_tax_rate(item, report.jurisdiction)
            analysis_results.append({
                "description": item.description,
                "commodity_type": validation_result.get('commodity_type', 'unknown'),
                "tax_rate": item.tax_rate,
                "recommended_rate": validation_result.get('recommended_tax_rate', Decimal('0.0')),
                "validation_status": validation_result.get('validation_status', 'UNKNOWN'),
                "confidence_score": validation_result.get('confidence_score', 0.0),
                "explanation": validation_result.get('explanation', ''),
                "is_acceptable": validation_result.get('is_acceptable', True)
            })
            
            if validation_result.get('is_acceptable', True):
                passed_validation += 1
        
        return {
            "total_items": total_items,
            "passed_validation": passed_validation,
            "failed_validation": total_items - passed_validation,
            "success_rate": passed_validation / total_items if total_items > 0 else 0.0,
            "item_analysis": analysis_results,
            "jurisdiction": report.jurisdiction,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }


# ---------- Currency Converter (unchanged except sqlite sync wrapped) ----------

class CurrencyConverter:
    def __init__(self, db_path=SQLITE_DB_PATH):
        self.db_path = db_path
        # We rely on DB initialization before usage, so no synchronous init needed here

    async def get_exchange_rate(self, from_currency: str, to_currency: str, date: datetime) -> Decimal:
        # Check cache (DB)
        rate = await run_db_query(self.get_exchange_rate_sync, from_currency, to_currency, date)
        if rate is None:
            # Fetch from API and store
            rate = await self.fetch_and_store_exchange_rate(from_currency, to_currency, date)
        return rate

    def get_exchange_rate_sync(self, from_currency: str, to_currency: str, date: datetime) -> Decimal:
        if from_currency == to_currency:
            return Decimal("1.0")
        date_str = date.date().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''SELECT rate FROM exchange_rates WHERE from_currency=? AND to_currency=? AND date=?''',
                      (from_currency, to_currency, date_str))
            row = c.fetchone()
            if row:
                return Decimal(str(row[0]))
        return None

    async def fetch_and_store_exchange_rate(self, from_currency: str, to_currency: str, date: datetime) -> Decimal:
        date_str = date.date().isoformat()
        url = f"https://api.exchangerate.host/{date_str}?base={from_currency}&symbols={to_currency}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    data = await resp.json()
                    rate = data.get("rates", {}).get(to_currency)
                    if rate:
                        rate_dec = Decimal(str(rate))
                        def store_rate():
                            with sqlite3.connect(self.db_path) as conn:
                                c = conn.cursor()
                                try:
                                    c.execute('''INSERT INTO exchange_rates (from_currency, to_currency, rate, date)
                                                 VALUES (?, ?, ?, ?)''',
                                              (from_currency, to_currency, float(rate_dec), date_str))
                                    conn.commit()
                                except sqlite3.IntegrityError:
                                    pass
                        await run_db_query(store_rate)
                        return rate_dec
        except Exception as e:
            print(f"Error fetching exchange rate: {e}")
        return None


def get_exchange_rate_sync(from_currency: str, to_currency: str, date: datetime) -> Decimal:
    if from_currency == to_currency:
        return Decimal("1.0")
    date_str = date.date().isoformat()
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''SELECT rate FROM exchange_rates WHERE from_currency=? AND to_currency=? AND date=?''',
                  (from_currency, to_currency, date_str))
        row = c.fetchone()
        if row:
            return Decimal(str(row[0]))
    return None  # Not found in DB

async def fetch_and_store_exchange_rate(from_currency: str, to_currency: str, date: datetime):
    date_str = date.date().isoformat()
    url = f"https://api.exchangerate.host/{date_str}?base={from_currency}&symbols={to_currency}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                data = await resp.json()
                rate = data.get("rates", {}).get(to_currency)
                if rate:
                    rate_dec = Decimal(str(rate))
                    # Store in DB
                    def store_rate():
                        with sqlite3.connect(SQLITE_DB_PATH) as conn:
                            c = conn.cursor()
                            try:
                                c.execute('''INSERT INTO exchange_rates (from_currency, to_currency, rate, date)
                                             VALUES (?, ?, ?, ?)''',
                                          (from_currency, to_currency, float(rate_dec), date_str))
                                conn.commit()
                            except sqlite3.IntegrityError:
                                pass
                    await run_db_query(store_rate)
                    return rate_dec
    except Exception as e:
        print(f"Error fetching exchange rate: {e}")
    return None

async def get_exchange_rate(from_currency: str, to_currency: str, date: datetime) -> Decimal:
    rate = await run_db_query(get_exchange_rate_sync, from_currency, to_currency, date)
    if rate is None:
        rate = await fetch_and_store_exchange_rate(from_currency, to_currency, date)
    return rate


# ---------- GPT-4 Category and Tax Validation (unchanged) ----------

def run_gpt4_call_sync(prompt: str):
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

async def gpt4_category_and_tax_validate(item: LineItem):
    prompt = f"""
You are a financial compliance expert and tax specialist.

Task: Classify the expense into ONE of these allowed categories exactly:
- meals, travel, accommodation, office_supplies, unknown

Then assess tax compliance at a high level.

Rules:
- Choose the best fitting category from the list above only. If none fit, use "unknown".
- Provide a short explanation.
- Provide a confidence score between 0.0 and 1.0 for the category classification.

Expense description: "{item.description}"
Amount: {item.total}
Tax rate: {item.tax_rate}
Jurisdiction: {item.jurisdiction}

Output valid JSON with keys:
- category: one of ["meals","travel","accommodation","office_supplies","unknown"]
- subcategory: string or null
- compliant: boolean
- explanation: string
- category_confidence: number between 0 and 1
"""
    loop = asyncio.get_running_loop()
    content = await loop.run_in_executor(None, run_gpt4_call_sync, prompt)
    try:
        return json.loads(content)
    except Exception:
        return {
            "category": item.category.value,
            "subcategory": None,
            "compliant": True,
            "explanation": "Failed to parse GPT response; assuming compliance.",
            "category_confidence": 0.0
        }


# ---------- Low-confidence Fallback Helpers ----------

def infer_category_keywords(description: str, vendor: str = None) -> str:
    """
    Heuristic keyword-based category inference. Returns one of the allowed categories
    ["meals", "travel", "accommodation", "office_supplies", "unknown"].
    """
    text = (description or "")
    vend = vendor or ""
    s = f"{text} {vend}".lower()

    # accommodation
    if any(k in s for k in ["hotel", "lodging", "motel", "inn", "resort", "accommodation", "stay"]):
        return "accommodation"

    # travel
    if any(k in s for k in [
        "uber", "ola", "taxi", "cab", "ride", "lyft", "transport", "transportation",
        "airfare", "flight", "airline", "train", "rail", "bus", "fare", "mileage", "fuel", "gas"
    ]):
        return "travel"

    # meals
    if any(k in s for k in [
        "restaurant", "lunch", "dinner", "breakfast", "meal", "meals", "cafe", "coffee", "snack", "food",
        "beverage", "catering"
    ]):
        return "meals"

    # office_supplies
    if any(k in s for k in [
        "office", "supplies", "paper", "pen", "pencil", "notebook", "notepad", "stapler", "envelope",
        "folder", "binder", "cartridge", "toner", "ink", "printer", "stationery"
    ]):
        return "office_supplies"

    return "unknown"


async def gpt4_category_reprompt(item: LineItem):
    """
    Second-pass categorization with few-shot examples and stricter constraints.
    """
    examples = [
        {"desc": "Lunch with client at Italian restaurant", "cat": "meals"},
        {"desc": "Uber ride to client office", "cat": "travel"},
        {"desc": "2 nights at Marriott Hotel", "cat": "accommodation"},
        {"desc": "Printer paper and ink cartridges", "cat": "office_supplies"},
    ]

    shots = "\n".join([f"- \"{e['desc']}\" -> {e['cat']}" for e in examples])

    prompt = f"""
You are a precise classifier. Map each expense to exactly one category from:
["meals","travel","accommodation","office_supplies","unknown"].
If none fits, return "unknown".

Few-shot guidance:
{shots}

Now classify this item:
Description: "{item.description}"
Vendor: "{item.vendor or ''}"
Jurisdiction: {item.jurisdiction}

Output valid JSON only with keys: category, subcategory, explanation, category_confidence.
"""
    loop = asyncio.get_running_loop()
    content = await loop.run_in_executor(None, run_gpt4_call_sync, prompt)
    try:
        data = json.loads(content)
        # Ensure proper fields
        cat = (data.get("category") or "unknown").lower()
        if cat not in {"meals", "travel", "accommodation", "office_supplies", "unknown"}:
            cat = "unknown"
        data["category"] = cat
        if "category_confidence" not in data:
            data["category_confidence"] = 0.0
        return data
    except Exception:
        return {
            "category": "unknown",
            "subcategory": None,
            "explanation": "Reprompt parse failure",
            "category_confidence": 0.0,
        }


# ---------- Financial Logic Agent (updated with async DB calls) ----------

class FinancialLogicAgent:
    def __init__(self, user: User):
        self.user = user
        self.access_control = AccessControl()
        self.tax_engine = GPTTaxValidationEngine()
        self.converter = CurrencyConverter()
        self.rounding_tolerance = Decimal("0.01")

    async def validate_expense_report(self, report: ExpenseReport) -> ValidationResult:
        violations = []
        warnings = []

        if not self.access_control.can_submit(self.user):
            violations.append(f"User role {self.user.role} not authorized to submit reports.")
            return ValidationResult(False, violations, warnings, 1.0, "REJECT")

        # Validate dates explicitly
        valid_dates, date_msg = validate_expense_dates(report.expense_date, report.submission_date)
        if not valid_dates:
            violations.append(date_msg)
            return ValidationResult(False, violations, warnings, 1.0, "REJECT")

        # Use GPT-4 to classify category and validate tax compliance per line item
        for item in report.line_items:
            try:
                vendor_name = item.vendor
                if vendor_name:
                    status, message = await check_vendor_policy(vendor_name, item.category.value, report.jurisdiction)
                    if status is False or status is None:
                        violations.append(message)

                exception_found, exception_reason = await check_exceptions(report.report_id, item.description)
                if exception_found:
                    warnings.append(f"Exception on '{item.description}': {exception_reason}")

                gpt_result = await gpt4_category_and_tax_validate(item)
                cat_str = gpt_result.get("category", item.category.value).lower()
                if cat_str in ExpenseCategory._value2member_map_:
                    item.category = ExpenseCategory(cat_str)
                compliant = gpt_result.get("compliant", True)
                explanation = gpt_result.get("explanation", "")
                cat_conf = float(gpt_result.get("category_confidence", 0.0) or 0.0)
                if not compliant:
                    violations.append(f"GPT flagged compliance issue on '{item.description}': {explanation}")

                # Low-confidence fallback: keyword rules + optional reprompt
                if cat_conf < 0.7:
                    inferred = infer_category_keywords(item.description, item.vendor)
                    if inferred != item.category.value:
                        warnings.append(
                            f"Low category confidence ({cat_conf:.2f}) for '{item.description}'. Heuristic -> {inferred}."
                        )
                        # Try a reprompt once for final decision
                        reprompt = await gpt4_category_reprompt(item)
                        final_cat = (reprompt.get("category") or inferred).lower()
                        if final_cat in ExpenseCategory._value2member_map_:
                            item.category = ExpenseCategory(final_cat)
                        else:
                            item.category = ExpenseCategory(inferred)
            except Exception as e:
                warnings.append(f"GPT category/tax validation error on '{item.description}': {str(e)}")

        # GPT-based tax rate validation for each line item
        for idx, item in enumerate(report.line_items, 1):
            # Use GPT to validate tax rate based on commodity identification
            tax_validation = await self.tax_engine.validate_tax_rate(item, report.jurisdiction)
            
            # Store commodity identification results on the line item for audit purposes
            item.commodity_type = tax_validation.get('commodity_type', 'unknown')
            item.tax_validation_status = tax_validation.get('validation_status', 'UNKNOWN')
            item.recommended_tax_rate = tax_validation.get('recommended_tax_rate', Decimal('0.0'))
            
            if not tax_validation.get('is_acceptable', True):
                violations.append(
                    f"Line item {idx} '{item.description}': Tax rate validation failed. "
                    f"Provided: {item.tax_rate}, Recommended: {tax_validation.get('recommended_tax_rate', 'Unknown')}. "
                    f"Reason: {tax_validation.get('explanation', 'No explanation provided')}"
                )
            
            # Log commodity identification for audit purposes
            commodity_type = tax_validation.get('commodity_type', 'unknown')
            confidence = tax_validation.get('confidence_score', 0.0)
            
            if confidence < 0.7:
                warnings.append(
                    f"Line item {idx} '{item.description}': Low confidence ({confidence:.2f}) "
                    f"in commodity identification as '{commodity_type}'"
                )

            calc_subtotal = decimal_round(item.quantity * item.unit_price)
            if abs(calc_subtotal - item.subtotal) > self.rounding_tolerance:
                violations.append(f"Line item {idx} subtotal mismatch")

            calc_tax = decimal_round(item.subtotal * item.tax_rate)
            if abs(calc_tax - item.tax_amount) > self.rounding_tolerance:
                violations.append(f"Line item {idx} tax amount mismatch")

        # Currency exchange validation
        for idx, item in enumerate(report.line_items, 1):
            if item.currency != report.currency:
                rate = await self.converter.get_exchange_rate(item.currency, report.currency, report.expense_date)
                if not rate:
                    violations.append(f"Line item {idx} missing exchange rate for {item.currency} to {report.currency}")

        # Validate claimed total matches calculated total after conversions
        total_calc = Decimal("0.0")
        for item in report.line_items:
            if item.currency == report.currency:
                total_calc += item.total
            else:
                rate = await self.converter.get_exchange_rate(item.currency, report.currency, report.expense_date)
                if rate:
                    total_calc += decimal_round(item.total * rate)

        # Approval threshold enforcement based on category
        for item in report.line_items:
            approval_threshold = await get_approval_threshold(item.category.value)
            if item.total > approval_threshold:
                violations.append(f"Line item '{item.description}' amount {item.total} exceeds approval threshold {approval_threshold} for category {item.category.value}.")

        if report.claimed_total and abs(total_calc - report.claimed_total) > self.rounding_tolerance:
            violations.append(f"Claimed total {report.claimed_total} mismatches calculated total {total_calc}")

        # Invoice-level mathematical validation (subtotal, tax, total)
        if report.invoice_metadata:
            inv = report.invoice_metadata or {}
            inv_subtotal = inv.get("subtotal")
            inv_tax = inv.get("total_tax_amount")
            inv_total = inv.get("total_amount")

            # Compute from line items in report currency
            computed_subtotal = Decimal("0.0")
            computed_tax = Decimal("0.0")
            for item in report.line_items:
                if item.currency == report.currency:
                    computed_subtotal += item.subtotal
                    computed_tax += item.tax_amount
                else:
                    rate = await self.converter.get_exchange_rate(item.currency, report.currency, report.expense_date)
                    if rate:
                        computed_subtotal += decimal_round(item.subtotal * rate)
                        computed_tax += decimal_round(item.tax_amount * rate)

            computed_total = decimal_round(computed_subtotal + computed_tax)

            # Compare if values provided
            if inv_subtotal is not None:
                try:
                    inv_subtotal_dec = Decimal(str(inv_subtotal))
                    if abs(inv_subtotal_dec - decimal_round(computed_subtotal)) > self.rounding_tolerance:
                        violations.append(
                            f"Invoice subtotal {inv_subtotal_dec} mismatches computed {decimal_round(computed_subtotal)}"
                        )
                except Exception:
                    warnings.append("Invoice subtotal not a valid number")

            if inv_tax is not None:
                try:
                    inv_tax_dec = Decimal(str(inv_tax))
                    if abs(inv_tax_dec - decimal_round(computed_tax)) > self.rounding_tolerance:
                        violations.append(
                            f"Invoice tax {inv_tax_dec} mismatches computed {decimal_round(computed_tax)}"
                        )
                except Exception:
                    warnings.append("Invoice total_tax_amount not a valid number")

            if inv_total is not None:
                try:
                    inv_total_dec = Decimal(str(inv_total))
                    if abs(inv_total_dec - computed_total) > self.rounding_tolerance:
                        violations.append(
                            f"Invoice total {inv_total_dec} mismatches computed {computed_total}"
                        )
                except Exception:
                    warnings.append("Invoice total_amount not a valid number")

        # Risk score calculation
        risk_score = 0.0
        if violations:
            risk_score += 0.5
        if self.user.role == Role.EMPLOYEE:
            risk_score += 0.1

        # Approval routing
        approval = ApprovalStatus.PENDING
        if self.access_control.can_approve(self.user, total_calc):
            if self.user.role == Role.MANAGER:
                approval = ApprovalStatus.MANAGER_APPROVED
            elif self.user.role == Role.DIRECTOR:
                approval = ApprovalStatus.DIRECTOR_APPROVED
            elif self.user.role == Role.CFO:
                approval = ApprovalStatus.CFO_APPROVED

        report.approval_status = approval
        report.risk_score = risk_score

        # Audit logging
        await log_audit(
            report.report_id,
            self.user.user_id,
            "VALIDATION",
            {
                "violations": violations,
                "warnings": warnings,
                "risk_score": risk_score,
                "approval_status": approval.value,
                "timestamp": datetime.utcnow().isoformat(),
                "commodity_analysis": {
                    item.description: {
                        "commodity_type": getattr(item, 'commodity_type', 'unknown'),
                        "tax_validation_status": getattr(item, 'tax_validation_status', 'unknown')
                    } for item in report.line_items
                }
            }
        )

        recommended_action = "REJECT" if violations else "AUTO_APPROVE"

        return ValidationResult(
            is_valid=(len(violations) == 0),
            violations=violations,
            warnings=warnings,
            risk_score=risk_score,
            recommended_action=recommended_action
        )


# ---------- JSON Parsing ----------

def parse_role(role_str):
    return Role(role_str.lower())

def parse_category(cat_str):
    cat_str = cat_str.lower()
    return ExpenseCategory(cat_str) if cat_str in ExpenseCategory._value2member_map_ else ExpenseCategory.UNKNOWN

def parse_user(data):
    return User(user_id=data["user_id"], role=parse_role(data["role"]))

def parse_line_item(data):
    return LineItem(
        description=data["description"],
        quantity=Decimal(str(data["quantity"])),
        unit_price=Decimal(str(data["unit_price"])),
        tax_rate=Decimal(str(data["tax_rate"])),
        category=parse_category(data.get("category", "unknown")),
        currency=data.get("currency", "USD"),
        jurisdiction=data.get("jurisdiction", "US"),
        vendor=data.get("vendor")
    )

def parse_expense_report(data):
    employee = parse_user(data["employee"])
    line_items = [parse_line_item(li) for li in data["line_items"]]
    claimed_total = Decimal(str(data["claimed_total"])) if "claimed_total" in data else None
    return ExpenseReport(
        report_id=data["report_id"],
        employee=employee,
        submission_date=datetime.fromisoformat(data["submission_date"]),
        expense_date=datetime.fromisoformat(data["expense_date"]),
        line_items=line_items,
        currency=data.get("currency", "USD"),
        jurisdiction=data.get("jurisdiction", "US"),
        claimed_total=claimed_total,
        invoice_metadata=data.get("invoice_metadata")
    )


# ---------- Main Execution ----------

async def main():
    await init_sqlite_db()

    with open("input.json", "r") as f:
        data = json.load(f)

    report = parse_expense_report(data)
    user = report.employee

    agent = FinancialLogicAgent(user)
    result = await agent.validate_expense_report(report)

    print("Validation Result:")
    print(f" Valid: {result.is_valid}")
    print(f" Violations: {result.violations}")
    print(f" Warnings: {result.warnings}")
    print(f" Risk Score: {result.risk_score}")
    print(f" Recommended Action: {result.recommended_action}")

    # Demonstrate GPT-based commodity analysis
    print("\n" + "="*50)
    print("GPT-BASED COMMODITY ANALYSIS")
    print("="*50)
    
    # Test API connectivity first
    print("Testing OpenAI API connectivity...")
    api_diagnostic = await agent.tax_engine.test_api_connectivity()
    
    print(f"API Key Configured: {api_diagnostic['api_key_configured']}")
    print(f"Client Initialized: {api_diagnostic['client_initialized']}")
    print(f"Model Name: {api_diagnostic['model_name']}")
    print(f"Test Result: {api_diagnostic['test_result']}")
    
    if api_diagnostic['error_message']:
        print(f"Error: {api_diagnostic['error_message']}")
        print("Using fallback commodity identification...")
    
    commodity_analysis = await agent.tax_engine.analyze_expense_report_commodities(report)
    
    print(f"Total Items Analyzed: {commodity_analysis['total_items']}")
    print(f"Tax Validation Success Rate: {commodity_analysis['success_rate']:.2%}")
    print(f"Items Passed Validation: {commodity_analysis['passed_validation']}")
    print(f"Items Failed Validation: {commodity_analysis['failed_validation']}")
    
    print("\nDetailed Item Analysis:")
    for i, item_analysis in enumerate(commodity_analysis['item_analysis'], 1):
        print(f"\nItem {i}: {item_analysis['description']}")
        print(f"  Commodity Type: {item_analysis['commodity_type']}")
        print(f"  Tax Rate: {item_analysis['tax_rate']} (Recommended: {item_analysis['recommended_rate']})")
        print(f"  Validation Status: {item_analysis['validation_status']}")
        print(f"  Confidence Score: {item_analysis['confidence_score']:.2f}")
        if item_analysis['explanation']:
            print(f"  Explanation: {item_analysis['explanation']}")


if __name__ == "__main__":
    asyncio.run(main())
