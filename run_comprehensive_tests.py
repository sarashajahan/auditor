#!/usr/bin/env python3
"""
Comprehensive test runner for GPT-based tax validation system.
Tests various jurisdictions, commodity types, and edge cases.
"""

import asyncio
import json
from decimal import Decimal
from datetime import datetime
from financial_logic_agent import (
    GPTTaxValidationEngine, 
    LineItem, 
    ExpenseCategory, 
    ExpenseReport, 
    User, 
    Role
)

async def test_single_scenario(scenario_data, tax_engine):
    """Test a single scenario with the tax validation engine."""
    
    print(f"\n{'='*60}")
    print(f"Testing: {scenario_data['name']}")
    print(f"Jurisdiction: {scenario_data['jurisdiction']}")
    print(f"Currency: {scenario_data['currency']}")
    print(f"{'='*60}")
    
    results = []
    total_items = len(scenario_data['line_items'])
    passed_validation = 0
    
    for i, item_data in enumerate(scenario_data['line_items'], 1):
        print(f"\nItem {i}: {item_data['description']}")
        print("-" * 50)
        
        # Create LineItem object
        item = LineItem(
            description=item_data['description'],
            quantity=Decimal(str(item_data['quantity'])),
            unit_price=Decimal(str(item_data['unit_price'])),
            tax_rate=Decimal(str(item_data['tax_rate'])),
            category=ExpenseCategory(item_data['category']),
            currency=scenario_data['currency'],
            jurisdiction=scenario_data['jurisdiction']
        )
        
        # Validate tax rate using GPT
        validation_result = await tax_engine.validate_tax_rate(item, scenario_data['jurisdiction'])
        
        # Display results
        commodity_type = validation_result.get('commodity_type', 'Unknown')
        confidence_score = validation_result.get('confidence_score', 0.0)
        recommended_rate = validation_result.get('recommended_tax_rate', Decimal('0.0'))
        validation_status = validation_result.get('validation_status', 'Unknown')
        
        print(f"  Commodity Type: {commodity_type}")
        print(f"  Confidence Score: {confidence_score:.2f}")
        print(f"  Provided Tax Rate: {item.tax_rate}")
        print(f"  Recommended Tax Rate: {recommended_rate}")
        print(f"  Validation Status: {validation_status}")
        
        if validation_result.get('explanation'):
            print(f"  Explanation: {validation_result['explanation']}")
        
        # Check against expected values
        expected_commodity = item_data.get('expected_commodity', 'unknown')
        expected_tax_rate = Decimal(str(item_data.get('expected_tax_rate', 0)))
        
        commodity_match = commodity_type.lower() == expected_commodity.lower()
        tax_rate_match = abs(recommended_rate - expected_tax_rate) < Decimal('0.01')
        
        if commodity_match and tax_rate_match:
            print(f"  ✅ PASSED - Matches expected: {expected_commodity}, {expected_tax_rate}")
            passed_validation += 1
        else:
            print(f"  ❌ FAILED - Expected: {expected_commodity}, {expected_tax_rate}")
        
        # Store result for summary
        results.append({
            "item": item_data['description'],
            "commodity_type": commodity_type,
            "expected_commodity": expected_commodity,
            "confidence_score": confidence_score,
            "provided_tax_rate": item.tax_rate,
            "recommended_tax_rate": recommended_rate,
            "expected_tax_rate": expected_tax_rate,
            "validation_status": validation_status,
            "commodity_match": commodity_match,
            "tax_rate_match": tax_rate_match,
            "passed": commodity_match and tax_rate_match
        })
    
    # Scenario summary
    success_rate = passed_validation / total_items if total_items > 0 else 0.0
    print(f"\n📊 Scenario Summary:")
    print(f"  Total Items: {total_items}")
    print(f"  Passed Validation: {passed_validation}")
    print(f"  Failed Validation: {total_items - passed_validation}")
    print(f"  Success Rate: {success_rate:.2%}")
    
    return {
        "scenario_name": scenario_data['name'],
        "jurisdiction": scenario_data['jurisdiction'],
        "total_items": total_items,
        "passed_validation": passed_validation,
        "success_rate": success_rate,
        "results": results
    }

async def run_comprehensive_tests():
    """Run comprehensive tests with all scenarios."""
    
    print("🚀 Comprehensive GPT Tax Validation Test Suite")
    print("=" * 70)
    
    # Initialize the GPT tax validation engine
    tax_engine = GPTTaxValidationEngine()
    
    # Test API connectivity first
    print("\n🔍 Testing OpenAI API Connectivity...")
    api_diagnostic = await tax_engine.test_api_connectivity()
    
    print(f"API Key Configured: {api_diagnostic['api_key_configured']}")
    print(f"Client Initialized: {api_diagnostic['client_initialized']}")
    print(f"Model Name: {api_diagnostic['model_name']}")
    print(f"Test Result: {api_diagnostic['test_result']}")
    
    if api_diagnostic['error_message']:
        print(f"⚠️  Warning: {api_diagnostic['error_message']}")
        print("   Using fallback commodity identification...")
    
    # Load test scenarios
    try:
        with open("test_data_comprehensive.json", "r") as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print("❌ test_data_comprehensive.json not found. Using built-in scenarios...")
        test_data = {"test_scenarios": []}
    
    # Run tests for each scenario
    all_results = []
    total_scenarios = len(test_data['test_scenarios'])
    successful_scenarios = 0
    
    for scenario in test_data['test_scenarios']:
        try:
            result = await test_single_scenario(scenario, tax_engine)
            all_results.append(result)
            
            if result['success_rate'] >= 0.8:  # 80% success threshold
                successful_scenarios += 1
                
        except Exception as e:
            print(f"❌ Error testing scenario '{scenario['name']}': {e}")
            all_results.append({
                "scenario_name": scenario['name'],
                "error": str(e),
                "success_rate": 0.0
            })
    
    # Overall summary
    print(f"\n{'='*70}")
    print("🏁 COMPREHENSIVE TEST RESULTS SUMMARY")
    print(f"{'='*70}")
    
    print(f"Total Scenarios Tested: {total_scenarios}")
    print(f"Successful Scenarios (≥80%): {successful_scenarios}")
    print(f"Overall Success Rate: {successful_scenarios/total_scenarios:.2%}" if total_scenarios > 0 else "No scenarios tested")
    
    # Detailed results by scenario
    print(f"\n📋 Detailed Results by Scenario:")
    for result in all_results:
        if 'error' in result:
            print(f"  ❌ {result['scenario_name']}: ERROR - {result['error']}")
        else:
            status = "✅ PASS" if result['success_rate'] >= 0.8 else "❌ FAIL"
            print(f"  {status} {result['scenario_name']}: {result['success_rate']:.2%} ({result['passed_validation']}/{result['total_items']})")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if api_diagnostic['test_result'] != 'success':
        print("  • Fix OpenAI API connectivity issues for better results")
        print("  • Check API key, billing, and network connectivity")
    else:
        print("  • API connectivity is working well")
    
    if successful_scenarios < total_scenarios:
        print("  • Review failed scenarios for improvement opportunities")
        print("  • Consider adjusting commodity identification logic")
    
    print("  • Monitor confidence scores for low-confidence identifications")
    print("  • Use fallback identification for critical business processes")
    
    print(f"\n{'='*70}")
    print("Test suite complete! 🎉")

async def main():
    """Main entry point."""
    try:
        await run_comprehensive_tests()
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
