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

def check_vendor_policy_sync(vendor_name, category, jurisdiction):
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT status, allowed_categories, regions FROM vendors WHERE vendor_name=?", (vendor_name,))
        row = c.fetchone()
    if not row:
        return None, f"Vendor '{vendor_name}' unknown, manual review recommended."
    status, allowed_cat_str, allowed_regions_str = row
    if status == "blacklisted":
        return False, f"Vendor '{vendor_name}' is blacklisted."
    try:
        allowed_categories = json.loads(allowed_cat_str)
    except Exception:
        allowed_categories = allowed_cat_str.split(",") if allowed_cat_str else []
    try:
        allowed_regions = json.loads(allowed_regions_str)
    except Exception:
        allowed_regions = allowed_regions_str.split(",") if allowed_regions_str else []
    if category not in allowed_categories:
        return False, f"Vendor '{vendor_name}' not authorized for category '{category}'."
    if jurisdiction not in allowed_regions:
        return False, f"Vendor '{vendor_name}' not authorized in region '{jurisdiction}'."
    return True, "Vendor compliant."

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
    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT max_approval_amount FROM approval_thresholds WHERE category=?", (category,))
        row = c.fetchone()
    if row and row[0] is not None:
        return Decimal(str(row[0]))  # Fixed: access row[0] instead of row
    return Decimal("0")


async def get_approval_threshold(category):
    return await run_db_query(get_approval_threshold_sync, category)


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


# ---------- Tax Rule Engine (unchanged) ----------

class TaxRuleEngine:
    tax_rates = {
        "US": {
            "meals": Decimal("0.10"),
            "travel": Decimal("0.08"),
            "office_supplies": Decimal("0.05"),
            "accommodation": Decimal("0.12"),
            "unknown": Decimal("0.1")
        },
        "EU": {
            "meals": Decimal("0.15"),
            "travel": Decimal("0.20"),
            "office_supplies": Decimal("0.07"),
            "accommodation": Decimal("0.18"),
            "unknown": Decimal("0.1")
        }
    }

    def get_tax_rate(self, jurisdiction: str, category: ExpenseCategory) -> Decimal:
        return self.tax_rates.get(jurisdiction, {}).get(category.value, Decimal("0.0"))


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

Given the details below about an expense, please:

1. Identify the most accurate main category and subcategory for this expense.
2. Validate if the tax rate and amount comply with relevant tax laws and corporate policies.
3. Include if the expense categorization and tax treatment are compliant or not.
4. Provide explanations for any compliance issues.

Expense description: "{item.description}"
Amount: {item.total}
Tax rate: {item.tax_rate}
Jurisdiction: {item.jurisdiction}

Output JSON with keys:
- category: string
- subcategory: string or null
- compliant: boolean
- explanation: string
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
            "explanation": "Failed to parse GPT response; assuming compliance."
        }


# ---------- Financial Logic Agent (updated with async DB calls) ----------

class FinancialLogicAgent:
    def __init__(self, user: User):
        self.user = user
        self.access_control = AccessControl()
        self.tax_engine = TaxRuleEngine()
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
                    if status is False:
                        violations.append(message)
                    elif status is None:
                        warnings.append(message)

                exception_found, exception_reason = await check_exceptions(report.report_id, item.description)
                if exception_found:
                    warnings.append(f"Exception on '{item.description}': {exception_reason}")

                gpt_result = await gpt4_category_and_tax_validate(item)
                cat_str = gpt_result.get("category", item.category.value).lower()
                if cat_str in ExpenseCategory._value2member_map_:
                    item.category = ExpenseCategory(cat_str)
                compliant = gpt_result.get("compliant", True)
                explanation = gpt_result.get("explanation", "")
                if not compliant:
                    violations.append(f"GPT flagged compliance issue on '{item.description}': {explanation}")
            except Exception as e:
                warnings.append(f"GPT category/tax validation error on '{item.description}': {str(e)}")

        # Deterministic tax and total validations
        for idx, item in enumerate(report.line_items, 1):
            expected_rate = self.tax_engine.get_tax_rate(report.jurisdiction, item.category)
            if expected_rate is None:
                violations.append(f"Unknown tax rate for jurisdiction {report.jurisdiction}, category {item.category.value}")
                continue
            if abs(item.tax_rate - expected_rate) > Decimal("0.005"):
                violations.append(f"Line item {idx} tax rate {item.tax_rate} deviates from expected {expected_rate}")

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
                "timestamp": datetime.utcnow().isoformat()
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
        claimed_total=claimed_total
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


if __name__ == "__main__":
    asyncio.run(main())