import requests
import json

def test_domain(domain):
    print(f"\nTesting domain: {domain}")
    response = requests.post(
        "http://localhost:8000/check_domain",
        json={"domain": domain}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("Status: Success")
        print(f"Is Phishing: {result['is_phishing']}")
        print(f"Risk Score: {result['risk_score']:.2f}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("\nSuspicious Features:")
        for feature in result['suspicious_features']:
            print(f"- {feature}")
        print("\nBrand Detection:")
        for brand, status in result['brand_detection'].items():
            print(f"- {brand}: {status}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # Test legitimate domains
    test_domain("google.com")
    test_domain("microsoft.com")
    test_domain("amazon.com")
    
    # Test suspicious domains
    test_domain("g00gle-secure.com")
    test_domain("paypal-verify-account.com")
    test_domain("login-microsoft-verify.com") 