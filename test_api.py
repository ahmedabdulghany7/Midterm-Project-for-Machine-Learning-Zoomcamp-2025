#!/usr/bin/env python
"""
Test script for the Concrete Strength Prediction API
This script tests all endpoints of the deployed Flask application.
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5000"


def test_health():
    """Test the health check endpoint."""
    print("\n" + "="*60)
    print("Testing Health Check Endpoint")
    print("="*60)

    response = requests.get(f"{BASE_URL}/health")

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'
    print("✓ Health check passed!")


def test_single_prediction():
    """Test single prediction endpoint."""
    print("\n" + "="*60)
    print("Testing Single Prediction Endpoint")
    print("="*60)

    # Test data (high strength concrete)
    data = {
        "cement": 540.0,
        "slag": 0.0,
        "fly_ash": 0.0,
        "water": 162.0,
        "superplasticizer": 2.5,
        "coarse_aggregate": 1040.0,
        "fine_aggregate": 676.0,
        "age": 28
    }

    print(f"Input data:")
    print(json.dumps(data, indent=2))

    response = requests.post(
        f"{BASE_URL}/predict",
        json=data,
        headers={'Content-Type': 'application/json'}
    )

    print(f"\nStatus Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))

    assert response.status_code == 200
    result = response.json()
    assert result['status'] == 'success'
    assert 'predicted_strength_MPa' in result
    print(f"\n✓ Predicted strength: {result['predicted_strength_MPa']:.2f} MPa")


def test_multiple_predictions():
    """Test multiple different concrete mixtures."""
    print("\n" + "="*60)
    print("Testing Multiple Predictions")
    print("="*60)

    test_cases = [
        {
            "name": "High Strength Concrete (28 days)",
            "data": {
                "cement": 540.0,
                "slag": 0.0,
                "fly_ash": 0.0,
                "water": 162.0,
                "superplasticizer": 2.5,
                "coarse_aggregate": 1040.0,
                "fine_aggregate": 676.0,
                "age": 28
            }
        },
        {
            "name": "Low Strength Concrete (3 days)",
            "data": {
                "cement": 200.0,
                "slag": 100.0,
                "fly_ash": 50.0,
                "water": 200.0,
                "superplasticizer": 0.0,
                "coarse_aggregate": 900.0,
                "fine_aggregate": 800.0,
                "age": 3
            }
        },
        {
            "name": "Medium Strength Concrete (90 days)",
            "data": {
                "cement": 300.0,
                "slag": 100.0,
                "fly_ash": 0.0,
                "water": 180.0,
                "superplasticizer": 5.0,
                "coarse_aggregate": 1000.0,
                "fine_aggregate": 750.0,
                "age": 90
            }
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print("-" * 60)

        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_case['data'],
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code == 200:
            result = response.json()
            strength = result['predicted_strength_MPa']
            print(f"✓ Predicted strength: {strength:.2f} MPa")
        else:
            print(f"✗ Error: {response.status_code}")
            print(response.text)

        time.sleep(0.5)  # Small delay between requests


def test_batch_prediction():
    """Test batch prediction endpoint."""
    print("\n" + "="*60)
    print("Testing Batch Prediction Endpoint")
    print("="*60)

    batch_data = {
        "samples": [
            {
                "cement": 540.0,
                "slag": 0.0,
                "fly_ash": 0.0,
                "water": 162.0,
                "superplasticizer": 2.5,
                "coarse_aggregate": 1040.0,
                "fine_aggregate": 676.0,
                "age": 28
            },
            {
                "cement": 300.0,
                "slag": 100.0,
                "fly_ash": 50.0,
                "water": 180.0,
                "superplasticizer": 5.0,
                "coarse_aggregate": 1000.0,
                "fine_aggregate": 750.0,
                "age": 28
            },
            {
                "cement": 450.0,
                "slag": 0.0,
                "fly_ash": 100.0,
                "water": 170.0,
                "superplasticizer": 8.0,
                "coarse_aggregate": 1020.0,
                "fine_aggregate": 700.0,
                "age": 90
            }
        ]
    }

    print(f"Sending {len(batch_data['samples'])} samples for batch prediction...")

    response = requests.post(
        f"{BASE_URL}/batch_predict",
        json=batch_data,
        headers={'Content-Type': 'application/json'}
    )

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Received {result['count']} predictions")
        print("\nPredictions:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"  Sample {i}: {pred:.2f} MPa")
    else:
        print(f"✗ Error: {response.text}")


def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\n" + "="*60)
    print("Testing Error Handling")
    print("="*60)

    # Test missing field
    print("\n1. Testing with missing field...")
    incomplete_data = {
        "cement": 540.0,
        "slag": 0.0,
        # Missing other fields
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=incomplete_data,
        headers={'Content-Type': 'application/json'}
    )

    print(f"Status Code: {response.status_code}")
    if response.status_code == 400:
        print("✓ Correctly returned 400 error for missing fields")
    else:
        print(f"Response: {response.json()}")

    # Test empty request
    print("\n2. Testing with empty request...")
    response = requests.post(
        f"{BASE_URL}/predict",
        json={},
        headers={'Content-Type': 'application/json'}
    )

    print(f"Status Code: {response.status_code}")
    if response.status_code == 400:
        print("✓ Correctly returned 400 error for empty request")


def test_performance():
    """Test API response time."""
    print("\n" + "="*60)
    print("Testing API Performance")
    print("="*60)

    data = {
        "cement": 540.0,
        "slag": 0.0,
        "fly_ash": 0.0,
        "water": 162.0,
        "superplasticizer": 2.5,
        "coarse_aggregate": 1040.0,
        "fine_aggregate": 676.0,
        "age": 28
    }

    num_requests = 10
    response_times = []

    print(f"Sending {num_requests} requests...")

    for i in range(num_requests):
        start_time = time.time()

        response = requests.post(
            f"{BASE_URL}/predict",
            json=data,
            headers={'Content-Type': 'application/json'}
        )

        end_time = time.time()
        response_time = (end_time - start_time) * 1000  # Convert to ms

        response_times.append(response_time)
        print(f"  Request {i+1}: {response_time:.2f} ms")

    avg_time = sum(response_times) / len(response_times)
    min_time = min(response_times)
    max_time = max(response_times)

    print(f"\nPerformance Statistics:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")

    if avg_time < 100:
        print("✓ Excellent performance (< 100ms)")
    elif avg_time < 500:
        print("✓ Good performance (< 500ms)")
    else:
        print("⚠ Performance could be improved")


def main():
    """Run all tests."""
    print("="*60)
    print("CONCRETE STRENGTH PREDICTION API - TEST SUITE")
    print("="*60)
    print(f"Testing API at: {BASE_URL}")

    try:
        # Run tests
        test_health()
        test_single_prediction()
        test_multiple_predictions()
        test_batch_prediction()
        test_error_handling()
        test_performance()

        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)

    except requests.exceptions.ConnectionError:
        print("\n" + "="*60)
        print("❌ ERROR: Cannot connect to API")
        print("="*60)
        print("\nPlease ensure the Flask server is running:")
        print("  python predict.py")
        print("\nOr update BASE_URL if deployed to a different location")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
