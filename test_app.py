import json
from app import app

def run_tests():
    client = app.test_client()
    passed = 0
    total = 0

    def assert_test(name, condition, details=""):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"[PASS] {name} {details}")
        else:
            print(f"[FAIL] {name} {details}")

    print("========================================")
    print("Testing GroundVision Flask Application")
    print("========================================")

    # 1. Test Web Page Routes
    pages = [
        ('/', 'Dashboard'),
        ('/dashboard', 'Dashboard Alias'),
        ('/prediction', 'Prediction Studio'),
        ('/analytics', 'Historical Analytics'),
        ('/insights', 'Groundwater Insights'),
        ('/reports', 'Reports & Exports'),
        ('/report', 'Report Alias')
    ]

    for route, label in pages:
        res = client.get(route)
        assert_test(f"Page GET {route} ({label})", res.status_code == 200, f"Status: {res.status_code}")

    # 2. Test API Overview
    res = client.get('/api/overview')
    assert_test("API /api/overview Status 200", res.status_code == 200)
    data = res.get_json()
    assert_test("API /api/overview Total Obs", data.get('total_observations') == 12349, f"Obs: {data.get('total_observations')}")
    assert_test("API /api/overview Total Districts", data.get('total_districts') == 33, f"Districts: {data.get('total_districts')}")

    # 3. Test API Districts
    res = client.get('/api/districts')
    assert_test("API /api/districts Status 200", res.status_code == 200)
    districts = res.get_json()
    assert_test("API /api/districts Count is 33", len(districts) == 33, f"Count: {len(districts)}")

    # 4. Test API Stations (Perambalur)
    res = client.get('/api/stations?district=Perambalur')
    assert_test("API /api/stations?district=Perambalur Status 200", res.status_code == 200)
    stations = res.get_json()
    assert_test("API /api/stations Perambalur has stations", len(stations) > 0, f"Wells: {len(stations)}")
    ammapalayam = next((s for s in stations if s['station'] == 'Ammapalayam'), None)
    assert_test("Ammapalayam found in Perambalur", ammapalayam is not None)

    # 5. Test API Station Details
    res = client.get('/api/station/Ammapalayam')
    assert_test("API /api/station/Ammapalayam Status 200", res.status_code == 200)
    st_data = res.get_json()
    assert_test("Ammapalayam readings history length", len(st_data.get('history', [])) >= 10, f"Readings: {len(st_data.get('history', []))}")

    # 6. Test API Predict (POST)
    payload = {
        'station': 'Ammapalayam',
        'horizon_days': 30,
        'rain_offset': -5,
        'extraction_demand': 'standard',
        'soil_moisture': 45
    }
    res = client.post('/api/predict', data=json.dumps(payload), content_type='application/json')
    assert_test("API POST /api/predict Status 200", res.status_code == 200)
    pred_data = res.get_json()
    assert_test("Prediction contains predicted_depth", 'predicted_depth' in pred_data, f"Depth: {pred_data.get('predicted_depth')}m")
    assert_test("Prediction contains trajectory steps", len(pred_data.get('trajectory', [])) > 0)

    # 7. Test API Analytics Data
    res = client.get('/api/analytics_data?station=Ammapalayam')
    assert_test("API /api/analytics_data Status 200", res.status_code == 200)
    analytics_data = res.get_json()
    metrics = analytics_data.get('metrics', {})
    assert_test("Held-out Test R2 score >= 0.85", metrics.get('test_r2', 0) >= 0.85, f"R2: {metrics.get('test_r2')}")
    assert_test("Test sample count is ~2470", metrics.get('test_samples') == 2470, f"Samples: {metrics.get('test_samples')}")
    assert_test("Hindcast records exist", len(analytics_data.get('hindcast', [])) > 0)

    # 8. Test API Insights Data
    res = client.get('/api/insights_data?district=Perambalur')
    assert_test("API /api/insights_data Status 200", res.status_code == 200)
    insights_data = res.get_json()
    assert_test("Deepest districts list populated", len(insights_data.get('deepest_districts', [])) == 5)
    assert_test("Shallowest districts list populated", len(insights_data.get('shallowest_districts', [])) == 5)

    # 9. Test API Export CSV
    res = client.get('/api/export/csv?station=Ammapalayam')
    assert_test("API /api/export/csv Status 200", res.status_code == 200)
    assert_test("Export CSV has text/csv mimetype", res.mimetype == 'text/csv')
    assert_test("Export CSV contains Station column", b'Station' in res.data)

    # 10. Test API Export Report
    res = client.get('/api/export/report?station=Ammapalayam')
    assert_test("API /api/export/report Status 200", res.status_code == 200)
    assert_test("Report contains GroundVision header", b'GroundVision' in res.data)
    assert_test("Report contains Ammapalayam station name", b'Ammapalayam' in res.data)

    print("========================================")
    print(f"Test Summary: {passed}/{total} Passed ({'ALL PASSED' if passed == total else 'SOME FAILED'})")
    print("========================================")

if __name__ == '__main__':
    run_tests()
