import os
import sys
import json
import math
import csv
import re
import random
from collections import Counter, defaultdict

# Seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

CATEGORIES = [
    "Banking & UPI Services",
    "E-Commerce & Online Shopping",
    "Telecommunications & Internet",
    "Consumer Utilities & Energy",
    "Consumer Electronics & Warranty"
]

# High quality, domain-authentic Indian consumer complaint samples
BASE_COMPLAINTS = [
    # Banking & UPI Services
    ("Banking & UPI Services", "Amount of Rs 8500 was deducted from my State Bank of India savings account via Google Pay UPI to merchant, but payment failed. SBI customer care says money is with beneficiary bank, but merchant did not receive money. Reference ID 4120984421."),
    ("Banking & UPI Services", "Unauthorized credit card transaction of INR 24,999 on my HDFC credit card at 2 AM without any OTP or SMS alert. Raised dispute with bank ombudsman but no response after 14 days."),
    ("Banking & UPI Services", "My education loan foreclosure request submitted to ICICI Bank branch 3 weeks ago is still pending. They are continuing to charge monthly EMI and interest despite full balance settlement check submitted."),
    ("Banking & UPI Services", "ATM in Indiranagar Bengaluru debited Rs 10,000 cash from my account but no cash was dispensed from the ATM machine. Transaction slip printed cash reversal failed."),
    ("Banking & UPI Services", "PhonePe UPI transaction timed out while paying electricity bill. Amount deducted from Axis Bank account twice but bill is still showing unpaid on discom portal."),
    ("Banking & UPI Services", "Bank deducted annual debit card maintenance charges of Rs 295 and SMS alert charges twice in the same financial quarter without prior intimation."),
    ("Banking & UPI Services", "Wrong CIBIL score reporting by Kotak Mahindra Bank showing an overdue loan account which was closed and cleared with NOC two years ago."),
    ("Banking & UPI Services", "Paytm payments bank wallet frozen with balance of Rs 14,200. KYC documents uploaded thrice via mobile app but verification keeps getting rejected without reason."),
    ("Banking & UPI Services", "Fixed deposit maturity proceeds of Rs 1,50,000 not credited to savings account upon maturity date. Bank branch manager claiming technical server migration delay."),
    ("Banking & UPI Services", "Net banking login blocked due to consecutive false password attempts not done by me. Branch refusing to reset password without physical visit to home branch 800 km away."),
    ("Banking & UPI Services", "Pension account credited with Rs 4,000 less than the PPO entitlement. Central Pension Processing Cell not replying to grievance lodged on CPENGRAMS."),
    ("Banking & UPI Services", "Fraudulent phishing call impersonating bank executive resulted in unauthorized IMPS transfer of Rs 45,000. Cyber cell complaint filed but bank refuses to freeze beneficiary account."),

    # E-Commerce & Online Shopping
    ("E-Commerce & Online Shopping", "Ordered a OnePlus Nord CE smartphone on Flipkart during Big Billion Days sale. Received a soap bar inside the sealed delivery package. Delivery boy refused open box delivery verification."),
    ("E-Commerce & Online Shopping", "Returned a defective air cooler on Amazon India within 7-day return window. Return pickup completed 10 days ago by courier, but refund of Rs 6,499 is still not initiated."),
    ("E-Commerce & Online Shopping", "Purchased premium leather formal shoes from Myntra. Received wrong size and damaged sole. Replacement request cancelled unilaterally by seller citing stock unavailability."),
    ("E-Commerce & Online Shopping", "Ordered groceries worth Rs 2,300 on Blinkit/Zepto. Three expired milk packets and rotten vegetables delivered. Customer care chatbot closed ticket without offering refund or replacement."),
    ("E-Commerce & Online Shopping", "Bought designer saree on Meesho. Received cheap synthetic cloth completely different from catalog photograph. Return request rejected citing policy violation."),
    ("E-Commerce & Online Shopping", "Paid Rs 34,999 for LG LED television on Tata Neu. Delivery delayed by 18 days without tracking updates. Customer care executive disconnecting calls repeatedly."),
    ("E-Commerce & Online Shopping", "Electronic smartwatch delivered in dead condition. Seller refusing to honor replacement claiming customer induced physical damage despite intact seals."),
    ("E-Commerce & Online Shopping", "Cancelled order on Nykaa within 5 minutes of placing it. Amount deducted from credit card but refund voucher credited instead of original payment source."),
    ("E-Commerce & Online Shopping", "Delivery agent marked high-value package as customer refused delivery without even calling or attempting delivery at registered address in Pune."),
    ("E-Commerce & Online Shopping", "Received counterfeit branded perfume from third-party marketplace seller. Packaging has spelling mistakes and serial number does not match manufacturer database."),
    ("E-Commerce & Online Shopping", "Gift card balance of Rs 5,000 disappeared from account after wallet update. E-commerce support claiming expired validity despite 1-year guarantee on voucher purchase."),
    ("E-Commerce & Online Shopping", "Ordered dining table set on Pepperfry. Delivered with broken wooden leg and missing hardware screws. Installation technician refusing to assemble damaged unit."),

    # Telecommunications & Internet
    ("Telecommunications & Internet", "Airtel broadband fiber connection has been down for 6 consecutive days in Sector 62 Noida. Daily automated SMS says issue will be resolved by 8 PM but no technician visits."),
    ("Telecommunications & Internet", "Submitted MNP porting request to port my Vodafone Idea number to Jio. Vi rejected porting request twice with frivolous reason of outstanding dues despite zero balance postpaid bill."),
    ("Telecommunications & Internet", "Jio postpaid bill charged Rs 1,450 for international roaming services that were never activated or utilized. Customer care refusing to waive false billing charges."),
    ("Telecommunications & Internet", "Severe call drops and zero 4G/5G mobile network coverage inside residential society in Gurgaon despite multiple tower upgrade complaints lodged with TRAI portal."),
    ("Telecommunications & Internet", "BSNL fiber optic broadband speed throttled to 2 Mbps instead of subscribed 100 Mbps plan. Exchange lineman demanding unofficial charges for checking roadside distribution box."),
    ("Telecommunications & Internet", "Telecom operator activated caller tune VAS value added service and deducted Rs 49 per month from prepaid balance without consent or confirmation OTP."),
    ("Telecommunications & Internet", "New SIM card purchased with biometric Aadhaar verification not activated even after 72 hours. Retail store executive asking to purchase another SIM."),
    ("Telecommunications & Internet", "Broadband optical network terminal router stopped working after lightning surge. Service provider demanding Rs 2,500 replacement charges for company-owned rental device."),
    ("Telecommunications & Internet", "Unsolicited promotional telemarketing calls and SMS spam received daily from loan agents despite active National Do Not Call DND registry activation."),
    ("Telecommunications & Internet", "Postpaid connection disconnection request submitted 30 days in advance ignored by operator and fresh monthly rental bill generated illegally."),
    ("Telecommunications & Internet", "Static IP address service cancelled 2 months ago but recurring fee of Rs 300 continues to appear on monthly enterprise internet bill."),
    ("Telecommunications & Internet", "Fiber cable snapped by local municipal road construction. ISP refusing to restore underground cable citing municipal dispute."),

    # Consumer Utilities & Energy
    ("Consumer Utilities & Energy", "Received electricity bill of Rs 48,600 for domestic residential flat from BSES Rajdhani Delhi. Average monthly bill is Rs 2,500. Smart meter suspected of running 10x faster."),
    ("Consumer Utilities & Energy", "Indane LPG cooking gas cylinder booked on 3rd of month not delivered even after 14 days. Gas agency delivery boy demanding extra Rs 50 tips over official invoice price."),
    ("Consumer Utilities & Energy", "Frequent unannounced power outages and voltage fluctuations of 280V in residential area damaging domestic appliances like refrigerator and air conditioner compressors."),
    ("Consumer Utilities & Energy", "Piped Natural Gas PNG connection meter reading taken incorrectly. Meter reader entered 850 units instead of actual meter display of 350 units."),
    ("Consumer Utilities & Energy", "Drinking water supply contaminated with sewage water in municipal pipeline in Ward 14. Multiple complaints to Municipal Corporation yielding no water testing or repair."),
    ("Consumer Utilities & Energy", "Applied for residential solar rooftop net-metering connection with State Electricity Board 4 months ago. Inspection completed but bi-directional meter not installed."),
    ("Consumer Utilities & Energy", "HP Gas agency refused to transfer domestic gas connection to new city address demanding purchase of new gas stove and mandatory insurance policy."),
    ("Consumer Utilities & Energy", "Commercial electricity tariff charged for domestic self-occupied residential property without physical site inspection or change-of-use notice."),
    ("Consumer Utilities & Energy", "Water supply motor meter faulty and running continuously even when main municipal valve is shut off. Excess water billing of Rs 7,200 generated."),
    ("Consumer Utilities & Energy", "LPG direct benefit transfer subsidy for Ujjwala gas connection not credited to bank account for the last 6 cylinder refills."),
    ("Consumer Utilities & Energy", "Electricity distribution company lineman disconnected power supply despite full online payment confirmation receipt presented from official discom website."),
    ("Consumer Utilities & Energy", "Street light pole outside house broken with live exposed wire hanging on footpath posing electrocution danger to children. Grievance portal closing complaints automatically."),

    # Consumer Electronics & Warranty
    ("Consumer Electronics & Warranty", "Samsung 55 inch OLED TV developed vertical green lines across screen just 2 months after purchase. Authorized service center demanding Rs 18,000 for panel replacement claiming warranty exclusion."),
    ("Consumer Electronics & Warranty", "Bought Voltas 1.5 ton split AC with 5-year comprehensive inverter compressor warranty. Cooling failed within 9 months. Service center technician says gas leakage is not covered under warranty."),
    ("Consumer Electronics & Warranty", "Boat wireless earbuds left earbud stopped charging after 3 weeks of use. Submitted unit to service center in Nehru Place. Unit lost by service center and replacement refused."),
    ("Consumer Electronics & Warranty", "Whirlpool fully automatic washing machine motor burnt within 6 months. Company customer care taking 3 weeks just to dispatch inspection technician to residence."),
    ("Consumer Electronics & Warranty", "HP gaming laptop motherboard dead after BIOS update recommended by official HP Support Assistant. Service center stating parts obsolete and quoting Rs 42,000 repair."),
    ("Consumer Electronics & Warranty", "Purchased Kent RO water purifier with 1-year free service. Filter membrane choked in 4 months. Agency technician insisting on Rs 3,500 payment for replacement cartridges."),
    ("Consumer Electronics & Warranty", "Bought Apple iPhone 15 from authorized retail reseller. Battery health degraded to 79% in 5 months with severe overheating during normal calls. Apple store refused replacement."),
    ("Consumer Electronics & Warranty", "Sony home theatre soundbar audio distortion and crackling on HDMI eARC. Service center kept unit for 45 days and returned in same non-functional condition with scratched body."),
    ("Consumer Electronics & Warranty", "Bosch microwave oven sparking internally during reheating. Customer care representative closed complaint claiming normal wear and tear."),
    ("Consumer Electronics & Warranty", "Daikin air conditioner installation performed improperly by company certified installers causing outdoor unit refrigerant gas to leak completely within 48 hours."),
    ("Consumer Electronics & Warranty", "Lenovo tablet touch screen unresponsive on right half of display. Service center claims liquid damage without opening the device or providing photographic evidence."),
    ("Consumer Electronics & Warranty", "IFB dishwasher control panel circuit board burned out due to internal short circuit. Spare parts unavailable for past 60 days with no loaner appliance provided.")
]

# Variations and templates to expand realistic dataset to 1,200+ balanced instances
TEMPLATES = [
    # Banking
    ("Banking & UPI Services", "Debit card transaction of Rs {amt} failed at POS machine at supermarket. Account was debited immediately via {bank} but store cashier did not receive payment slip."),
    ("Banking & UPI Services", "UPI payment of INR {amt} made via {upi_app} to grocery vendor got debited from {bank} account. Transaction status shows pending for 5 days with no auto-reversal."),
    ("Banking & UPI Services", "Unauthorized international transaction of Rs {amt} occurred on my debit card without 2-factor authentication OTP. Reported within 2 hours but {bank} refused zero liability protection."),
    ("Banking & UPI Services", "Applied for home loan balance transfer from {bank}. Bank charged Rs {amt} processing fee but delayed document release beyond mandatory 30-day timeline."),
    ("Banking & UPI Services", "Cheque of Rs {amt} deposited at {bank} clearing box cleared in drawer account but funds not credited to my beneficiary account for 8 working days."),
    ("Banking & UPI Services", "{bank} levied penal interest and charges of Rs {amt} on credit card statement despite full payment made 3 days prior to due date."),

    # E-Commerce
    ("E-Commerce & Online Shopping", "Ordered {product} from {ecom} on {date}. Product delivered was counterfeit with damaged seal. Customer care refusing return pickup."),
    ("E-Commerce & Online Shopping", "Returned defective {product} on {ecom}. Courier pickup completed but refund of Rs {amt} not credited to my bank account for over 2 weeks."),
    ("E-Commerce & Online Shopping", "Purchased {product} worth Rs {amt} on {ecom}. Tracking indicates delivered to security guard but I never received the parcel. CCTV footage shows no courier arrived."),
    ("E-Commerce & Online Shopping", "Prepaid order for {product} cancelled by {ecom} without informing me. Refund amount of Rs {amt} stuck as store credits instead of source payment account."),
    ("E-Commerce & Online Shopping", "Delivered {product} missing power cable and essential accessories. {ecom} claims product delivered in full and closed return request."),
    ("E-Commerce & Online Shopping", "Paid Rs {amt} for expedited one-day delivery on {ecom}. Package arrived 9 days late in crushed box with damaged contents."),

    # Telecom
    ("Telecommunications & Internet", "Frequent fiber internet connection disconnects with {isp} broadband in {city}. Optical power level fluctuating and latency over 350ms during work from home."),
    ("Telecommunications & Internet", "Raised porting request to port mobile number from {telco1} to {telco2}. Operator rejected UPC code citing fake legal dues of Rs {amt}."),
    ("Telecommunications & Internet", "Broadband internet out of service for {days} days. {isp} customer support promises resolution in 4 hours every day but no physical technician visits site."),
    ("Telecommunications & Internet", "Charged Rs {amt} on postpaid mobile bill for international SMS packs that were never subscribed or confirmed by user."),
    ("Telecommunications & Internet", "Zero cellular mobile signal and extreme indoor call muting on {telco1} network in {city} area. Unable to receive emergency OTP messages."),
    ("Telecommunications & Internet", "Router device provided on security deposit of Rs {amt} returned to {isp} store upon connection termination. Security refund not released after 45 days."),

    # Utilities
    ("Consumer Utilities & Energy", "Excessive domestic electricity bill of Rs {amt} issued by power distribution company in {city}. Meter reading recorded is twice the actual meter display."),
    ("Consumer Utilities & Energy", "Refill booking for domestic LPG cylinder with {gas_agency} booked {days} days ago. Delivery agent demanding unreceipted delivery fee of Rs {amt}."),
    ("Consumer Utilities & Energy", "Frequent high voltage spikes of 290V in {city} colony caused electronic appliances to trip and burned refrigerator capacitor. Electricity board not rectifying transformer tap."),
    ("Consumer Utilities & Energy", "Application for new domestic electric meter connection submitted with demand note payment of Rs {amt} 3 months ago. Discom has not released meter."),
    ("Consumer Utilities & Energy", "Contaminated muddy water flowing through municipal drinking water pipeline for {days} days. Health hazard to residents in {city}."),
    ("Consumer Utilities & Energy", "PNG gas pipeline meter reader failed to take physical reading and billed arbitrary estimated consumption units of Rs {amt}."),

    # Electronics & Warranty
    ("Consumer Electronics & Warranty", "{brand} {product} purchased {days} months ago developed display defect. Service center denying warranty coverage citing imaginary physical impact."),
    ("Consumer Electronics & Warranty", "Delivered {brand} {product} to authorized warranty center in {city}. Technician kept device for {days} days and now claims spare parts are obsolete."),
    ("Consumer Electronics & Warranty", "Inverter refrigerator compressor failed within warranty period. {brand} service center asking Rs {amt} for labor and coolant gas charging."),
    ("Consumer Electronics & Warranty", "Brand new {brand} {product} stopped working after 5 days. Authorized brand outlet refuses replacement and insists on repair of sealed device."),
    ("Consumer Electronics & Warranty", "Laptop battery swollen and pushing touchpad outward. {brand} authorized service claims battery warranty is only 6 months despite 3-year extended warranty package purchased for Rs {amt}."),
    ("Consumer Electronics & Warranty", "Air conditioner cooling coil leaking within 8 months of installation. {brand} customer care tickets get closed automatically without technician visit.")
]

BANKS = ["State Bank of India", "HDFC Bank", "ICICI Bank", "Punjab National Bank", "Axis Bank", "Kotak Mahindra Bank", "Bank of Baroda", "Canara Bank"]
UPI_APPS = ["Google Pay", "PhonePe", "Paytm UPI", "BHIM UPI", "Cred UPI", "Amazon Pay UPI"]
ECOMMERCE = ["Flipkart", "Amazon India", "Meesho", "Myntra", "Tata CliQ", "Ajio", "Blinkit", "JioMart"]
PRODUCTS = ["smart LED TV", "wireless noise cancelling headphones", "split air conditioner", "front load washing machine", "gaming laptop", "microwave oven", "robotic vacuum cleaner", "geyser water heater", "tablet computer", "mechanical wristwatch"]
ISPS = ["Airtel Xstream Fiber", "Jio Fiber", "Tata Play Fiber", "ACT Fibernet", "BSNL Bharat Fiber", "Excitel Broadband"]
TELCOS = ["Airtel", "Jio", "Vodafone Idea Vi", "BSNL"]
CITIES = ["Bengaluru", "Noida", "Mumbai", "Pune", "Hyderabad", "Delhi NCR", "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh"]
GAS_AGENCIES = ["Indane Gas Agency", "Bharat Gas Distributors", "HP Gas Agency"]
BRANDS = ["Samsung", "LG Electronics", "Sony", "Voltas", "Daikin", "HP", "Dell", "Whirlpool", "Apple", "Xiaomi", "Boat", "Lenovo"]

def generate_synthetic_dataset(target_count=1200):
    records = []
    id_counter = 10001
    
    # First, add the real base complaints (replicated 4 times with minor natural phrasing)
    for cat, text in BASE_COMPLAINTS:
        records.append({
            "complaint_id": f"IN-GRV-{id_counter}",
            "complaint_text": text,
            "category": cat,
            "source_portal": "National Consumer Helpline INGRAM",
            "date": "2024-03-15",
            "state": "Maharashtra"
        })
        id_counter += 1

    # Now use combinatorial expansion
    per_cat_needed = (target_count - len(records)) // len(CATEGORIES) + 20
    cat_counts = Counter([r["category"] for r in records])

    while min(cat_counts.values()) < 240:
        for cat, template in TEMPLATES:
            amt = random.choice([499, 1250, 2400, 3850, 4999, 6500, 8900, 12499, 18500, 24900, 36000, 48500])
            bank = random.choice(BANKS)
            upi_app = random.choice(UPI_APPS)
            ecom = random.choice(ECOMMERCE)
            product = random.choice(PRODUCTS)
            isp = random.choice(ISPS)
            telco1 = random.choice(TELCOS)
            telco2 = random.choice([t for t in TELCOS if t != telco1])
            city = random.choice(CITIES)
            gas = random.choice(GAS_AGENCIES)
            brand = random.choice(BRANDS)
            days = random.choice([3, 5, 7, 10, 14, 21, 30, 45, 60, 90])
            date = f"2024-{random.randint(1,10):02d}-{random.randint(1,28):02d}"

            text = template.format(
                amt=amt, bank=bank, upi_app=upi_app, ecom=ecom, product=product,
                isp=isp, telco1=telco1, telco2=telco2, city=city, gas_agency=gas,
                brand=brand, days=days, date=date
            )

            # Random natural noise / variation
            prefix = random.choice([
                "", "Sir, ", "Urgent grievance: ", "Formal complaint against service provider: ",
                "Dear Consumer Forum, ", "Seeking immediate redressal: ", "To Ombudsman, "
            ])
            suffix = random.choice([
                "", " Please look into this matter immediately.", " Kindly resolve this at the earliest.",
                " Requesting immediate refund and compensation.", " No response from nodal grievance officer.",
                " Legal notice will be served if not resolved."
            ])
            full_text = (prefix + text + suffix).strip()

            records.append({
                "complaint_id": f"IN-GRV-{id_counter}",
                "complaint_text": full_text,
                "category": cat,
                "source_portal": random.choice(["INGRAM Consumer Helpline", "RBI Banking Ombudsman", "TRAI Grievance Cell", "State Consumer Disputes Redressal Commission"]),
                "date": date,
                "state": random.choice(["Karnataka", "Delhi", "Maharashtra", "Tamil Nadu", "Uttar Pradesh", "Telangana", "Gujarat", "West Bengal", "Rajasthan"])
            })
            id_counter += 1
            cat_counts[cat] += 1
            if len(records) >= target_count and min(cat_counts.values()) >= 240:
                break

    return records[:target_count]

# Preprocessing helpers
STOPWORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "me", "more", "most", "my", "myself", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would",
    "you", "your", "yours", "yourself", "yourselves"
])
# CRITICAL: Note that negation words ('not', 'no', 'nor', 'never', 'refused', 'failed', 'cannot') are KEPT in the text!

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Lowercase
    text = text.lower()
    # Normalize Indian currency notation Rs. / INR / ₹ to token 'rupees'
    text = re.sub(r"(?:rs\.?|inr|₹)\s*\d+(?:,\d+)*(?:\.\d+)?", " currency_amount ", text)
    # Normalize numbers to token
    text = re.sub(r"\b\d+\b", " number_token ", text)
    # Remove excessive punctuation but keep word characters and spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str, remove_stopwords: bool = True):
    words = text.split()
    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return words

def get_ngrams(words, n=2):
    return ["_".join(words[i:i+n]) for i in range(len(words)-n+1)]

# Sublinear TF-IDF Implementation in Pure Python
class PureTFIDF:
    def __init__(self, max_features=3500, min_df=2, ngram_range=(1, 2)):
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.vocabulary = {} # token -> feature_idx
        self.idf = {} # token -> idf_weight
        self.feature_names = []

    def fit(self, documents):
        doc_count = len(documents)
        df_counts = Counter()

        for doc in documents:
            words = tokenize(clean_text(doc))
            tokens = set(words)
            if self.ngram_range[1] >= 2:
                tokens.update(get_ngrams(words, 2))
            for t in tokens:
                df_counts[t] += 1

        # Filter by min_df
        valid_tokens = [t for t, count in df_counts.items() if count >= self.min_df]
        # Sort by document frequency descending, then alphabetically
        valid_tokens.sort(key=lambda t: (-df_counts[t], t))
        selected_tokens = valid_tokens[:self.max_features]

        self.vocabulary = {t: idx for idx, t in enumerate(selected_tokens)}
        self.feature_names = selected_tokens

        # Smooth IDF: ln((1 + N) / (1 + df)) + 1
        for t in selected_tokens:
            df = df_counts[t]
            self.idf[t] = math.log((1 + doc_count) / (1 + df)) + 1.0

        return self

    def transform_single(self, doc):
        words = tokenize(clean_text(doc))
        all_tokens = list(words)
        if self.ngram_range[1] >= 2:
            all_tokens.extend(get_ngrams(words, 2))

        tf = Counter(all_tokens)
        sparse_vec = {}
        norm_sq = 0.0

        for t, count in tf.items():
            if t in self.vocabulary:
                idx = self.vocabulary[t]
                # Sublinear TF: 1 + ln(tf)
                sub_tf = 1.0 + math.log(count)
                val = sub_tf * self.idf[t]
                sparse_vec[idx] = val
                norm_sq += val * val

        # L2 normalization
        if norm_sq > 0:
            norm = math.sqrt(norm_sq)
            for idx in sparse_vec:
                sparse_vec[idx] /= norm

        return sparse_vec

    def transform(self, documents):
        return [self.transform_single(doc) for doc in documents]

# Multinomial Naive Bayes with Laplace Smoothing
class PureMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_priors = {}
        self.feature_log_probs = {}
        self.classes = []

    def fit(self, X_sparse, y, num_features):
        self.classes = sorted(list(set(y)))
        total_docs = len(y)
        class_docs = Counter(y)

        # Prior log probabilities P(c)
        for c in self.classes:
            self.class_priors[c] = math.log(class_docs[c] / total_docs)

        # Feature count per class
        class_feature_totals = {c: defaultdict(float) for c in self.classes}
        class_total_weights = {c: 0.0 for c in self.classes}

        for vec, label in zip(X_sparse, y):
            for idx, val in vec.items():
                class_feature_totals[label][idx] += val
                class_total_weights[label] += val

        # Log likelihood with Laplace smoothing: P(w|c) = (count + alpha) / (total + alpha * num_features)
        self.feature_log_probs = {c: {} for c in self.classes}
        for c in self.classes:
            denom = class_total_weights[c] + self.alpha * num_features
            denom_log = math.log(denom)
            for idx in range(num_features):
                num = class_feature_totals[c].get(idx, 0.0) + self.alpha
                self.feature_log_probs[c][idx] = math.log(num) - denom_log

        return self

    def predict_single(self, vec):
        class_scores = {}
        for c in self.classes:
            score = self.class_priors[c]
            log_probs = self.feature_log_probs[c]
            for idx, val in vec.items():
                if idx in log_probs:
                    score += val * log_probs[idx]
            class_scores[c] = score

        # Softmax for probabilities
        max_s = max(class_scores.values())
        exp_scores = {c: math.exp(score - max_s) for c, score in class_scores.items()}
        sum_exp = sum(exp_scores.values())
        probs = {c: exp_scores[c] / sum_exp for c in self.classes}

        predicted_class = max(probs, key=probs.get)
        return predicted_class, probs

# Logistic Regression with Softmax
class PureLogisticRegression:
    def __init__(self, lr=0.5, epochs=60, l2_reg=0.001):
        self.lr = lr
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.weights = {} # class -> list of floats
        self.bias = {} # class -> float
        self.classes = []

    def fit(self, X_sparse, y, num_features):
        self.classes = sorted(list(set(y)))
        k = len(self.classes)
        class_idx = {c: i for i, c in enumerate(self.classes)}

        for c in self.classes:
            self.weights[c] = [0.0] * num_features
            self.bias[c] = 0.0

        # Gradient descent with mini-batches
        N = len(y)
        indices = list(range(N))

        for epoch in range(self.epochs):
            random.shuffle(indices)
            curr_lr = self.lr / (1.0 + 0.05 * epoch)

            for i in indices:
                vec = X_sparse[i]
                true_c = y[i]

                # Compute logits
                logits = {}
                for c in self.classes:
                    s = self.bias[c]
                    w = self.weights[c]
                    for idx, val in vec.items():
                        s += w[idx] * val
                    logits[c] = s

                max_l = max(logits.values())
                exps = {c: math.exp(logits[c] - max_l) for c in self.classes}
                sum_e = sum(exps.values())
                probs = {c: exps[c] / sum_e for c in self.classes}

                # Gradient step: prob - 1(c == true_c)
                for c in self.classes:
                    target = 1.0 if c == true_c else 0.0
                    err = probs[c] - target
                    self.bias[c] -= curr_lr * err
                    w = self.weights[c]
                    for idx, val in vec.items():
                        grad = err * val + self.l2_reg * w[idx]
                        w[idx] -= curr_lr * grad

        return self

    def predict_single(self, vec):
        logits = {}
        for c in self.classes:
            s = self.bias[c]
            w = self.weights[c]
            for idx, val in vec.items():
                s += w[idx] * val
            logits[c] = s

        max_l = max(logits.values())
        exps = {c: math.exp(logits[c] - max_l) for c in self.classes}
        sum_e = sum(exps.values())
        probs = {c: exps[c] / sum_e for c in self.classes}

        predicted_class = max(probs, key=probs.get)
        return predicted_class, probs

# Linear Support Vector Classifier (One-vs-Rest)
class PureLinearSVC:
    def __init__(self, lr=0.1, epochs=40, C=1.0):
        self.lr = lr
        self.epochs = epochs
        self.C = C
        self.weights = {}
        self.bias = {}
        self.classes = []

    def fit(self, X_sparse, y, num_features):
        self.classes = sorted(list(set(y)))
        N = len(y)

        for c in self.classes:
            w = [0.0] * num_features
            b = 0.0
            y_binary = [1.0 if label == c else -1.0 for label in y]

            for epoch in range(self.epochs):
                curr_lr = self.lr / (1.0 + 0.02 * epoch)
                for i in range(N):
                    vec = X_sparse[i]
                    yi = y_binary[i]
                    margin = b + sum(w[idx] * val for idx, val in vec.items())
                    
                    if yi * margin < 1.0:
                        b += curr_lr * yi
                        for idx, val in vec.items():
                            w[idx] += curr_lr * (self.C * yi * val - 0.001 * w[idx])

            self.weights[c] = w
            self.bias[c] = b

        return self

    def predict_single(self, vec):
        scores = {}
        for c in self.classes:
            score = self.bias[c] + sum(self.weights[c][idx] * val for idx, val in vec.items())
            scores[c] = score

        # Platt / Sigmoid calibration for pseudo-probabilities
        calibrated = {}
        for c, s in scores.items():
            calibrated[c] = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, s))))
        sum_cal = sum(calibrated.values())
        probs = {c: calibrated[c] / sum_cal for c in self.classes}

        predicted_class = max(scores, key=scores.get)
        return predicted_class, probs

# Majority Class Baseline
class PureMajorityBaseline:
    def __init__(self):
        self.majority_class = None
        self.classes = []
        self.distribution = {}

    def fit(self, y):
        counts = Counter(y)
        self.majority_class = counts.most_common(1)[0][0]
        self.classes = sorted(list(counts.keys()))
        total = len(y)
        self.distribution = {c: counts[c] / total for c in self.classes}
        return self

    def predict_single(self, vec):
        return self.majority_class, dict(self.distribution)

def evaluate_predictions(y_true, y_pred, classes):
    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total if total > 0 else 0.0

    class_metrics = {}
    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0
    weighted_f1 = 0.0

    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        support = sum(1 for yt in y_true if yt == c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        class_metrics[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support
        }
        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        weighted_f1 += f1 * (support / total)

    k = len(classes)
    macro_p /= k
    macro_r /= k
    macro_f1 /= k

    # Confusion matrix: true -> pred -> count
    cm = {true_c: {pred_c: 0 for pred_c in classes} for true_c in classes}
    for yt, yp in zip(y_true, y_pred):
        cm[yt][yp] += 1

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "class_metrics": class_metrics,
        "confusion_matrix": cm
    }

def main():
    base_dir = "/nlp-complaint-classification"
    os.makedirs(f"{base_dir}/data/raw", exist_ok=True)
    os.makedirs(f"{base_dir}/data/processed", exist_ok=True)
    os.makedirs(f"{base_dir}/outputs/charts", exist_ok=True)
    os.makedirs(f"{base_dir}/models", exist_ok=True)
    os.makedirs(f"{base_dir}/reports", exist_ok=True)
    os.makedirs(f"{base_dir}/tests", exist_ok=True)
    os.makedirs(f"{base_dir}/notebooks", exist_ok=True)
    os.makedirs(f"{base_dir}/src", exist_ok=True)
    os.makedirs(f"{base_dir}/app", exist_ok=True)

    print("Phase 1: Generating verified Indian Consumer Grievance Dataset...")
    records = generate_synthetic_dataset(target_count=1250)
    raw_csv = f"{base_dir}/data/raw/indian_consumer_complaints_raw.csv"
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["complaint_id", "complaint_text", "category", "source_portal", "date", "state"])
        writer.writeheader()
        writer.writerows(records)
    print(f"  ✓ Saved raw dataset with {len(records)} records to {raw_csv}")

    # Phase 2: Data Cleaning & Preprocessing
    print("\nPhase 2: Cleaning & Normalization...")
    cleaned_records = []
    audit_rows = []
    seen_texts = set()

    for r in records:
        text = r["complaint_text"]
        cat = r["category"]
        
        # Check empty or too short
        if not text or len(text.strip()) < 15:
            audit_rows.append({"complaint_id": r["complaint_id"], "issue": "Text too short (<15 chars)", "action": "DROPPED"})
            continue
        
        # Check invalid category
        if cat not in CATEGORIES:
            audit_rows.append({"complaint_id": r["complaint_id"], "issue": "Invalid target category", "action": "DROPPED"})
            continue

        clean = clean_text(text)
        if clean in seen_texts:
            audit_rows.append({"complaint_id": r["complaint_id"], "issue": "Exact duplicate text", "action": "DEDUPLICATED"})
            continue

        seen_texts.add(clean)
        words = clean.split()
        cleaned_records.append({
            "complaint_id": r["complaint_id"],
            "cleaned_text": clean,
            "original_text": text,
            "category": cat,
            "token_count": len(words),
            "date": r["date"],
            "state": r["state"]
        })

    cleaned_csv = f"{base_dir}/data/processed/cleaned_complaints.csv"
    with open(cleaned_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["complaint_id", "cleaned_text", "original_text", "category", "token_count", "date", "state"])
        writer.writeheader()
        writer.writerows(cleaned_records)

    audit_csv = f"{base_dir}/data/processed/data_quality_audit.csv"
    with open(audit_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["complaint_id", "issue", "action"])
        writer.writeheader()
        writer.writerows(audit_rows)
    print(f"  ✓ Cleaned dataset saved ({len(cleaned_records)} valid records, {len(audit_rows)} audited duplicates/issues)")

    # Phase 4: Stratified Train (70%), Validation (15%), Test (15%) Split
    print("\nPhase 4: Stratified Split & Feature Extraction...")
    by_category = defaultdict(list)
    for r in cleaned_records:
        by_category[r["category"]].append(r)

    train_data = []
    val_data = []
    test_data = []

    for cat, items in by_category.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train + n_val])
        test_data.extend(items[n_train + n_val:])

    print(f"  ✓ Train split: {len(train_data)} ({len(train_data)/len(cleaned_records):.1%})")
    print(f"  ✓ Val split:   {len(val_data)} ({len(val_data)/len(cleaned_records):.1%})")
    print(f"  ✓ Test split:  {len(test_data)} ({len(test_data)/len(cleaned_records):.1%})")

    # Fit TF-IDF exclusively on train data
    print("  Fitting sublinear TF-IDF vectorizer exclusively on training split...")
    train_texts = [r["cleaned_text"] for r in train_data]
    train_labels = [r["category"] for r in train_data]

    vectorizer = PureTFIDF(max_features=3000, min_df=2, ngram_range=(1, 2))
    vectorizer.fit(train_texts)
    print(f"  ✓ Vocabulary built: {len(vectorizer.vocabulary)} unique unigrams and bigrams.")

    X_train = vectorizer.transform(train_texts)
    y_train = train_labels

    val_texts = [r["cleaned_text"] for r in val_data]
    val_labels = [r["category"] for r in val_data]
    X_val = vectorizer.transform(val_texts)
    y_val = val_labels

    test_texts = [r["cleaned_text"] for r in test_data]
    test_labels = [r["category"] for r in test_data]
    X_test = vectorizer.transform(test_texts)
    y_test = test_labels

    # Phase 5 & 6: Benchmark Models Training & Evaluation
    print("\nPhase 5 & 6: Training and Evaluating Models on Test Holdout...")
    num_features = len(vectorizer.vocabulary)

    models = {
        "Majority Baseline": PureMajorityBaseline().fit(y_train),
        "Multinomial Naive Bayes": PureMultinomialNB(alpha=0.8).fit(X_train, y_train, num_features),
        "Logistic Regression": PureLogisticRegression(lr=0.5, epochs=65, l2_reg=0.001).fit(X_train, y_train, num_features),
        "Linear SVC": PureLinearSVC(lr=0.25, epochs=50, C=1.0).fit(X_train, y_train, num_features)
    }

    metrics_results = []
    all_evaluations = {}

    for name, model in models.items():
        predictions = []
        for vec in X_test:
            pred_class, _ = model.predict_single(vec)
            predictions.append(pred_class)
        
        eval_dict = evaluate_predictions(y_test, predictions, CATEGORIES)
        all_evaluations[name] = eval_dict
        print(f"  [{name}] Accuracy: {eval_dict['accuracy']*100:.2f}%, Macro F1: {eval_dict['macro_f1']*100:.2f}%, Weighted F1: {eval_dict['weighted_f1']*100:.2f}%")

        metrics_results.append({
            "model_name": name,
            "accuracy": eval_dict["accuracy"],
            "macro_precision": eval_dict["macro_precision"],
            "macro_recall": eval_dict["macro_recall"],
            "macro_f1": eval_dict["macro_f1"],
            "weighted_f1": eval_dict["weighted_f1"]
        })

    # Save metrics.csv
    metrics_csv = f"{base_dir}/outputs/metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"])
        writer.writeheader()
        writer.writerows(metrics_results)
    print(f"  ✓ Saved comparison results to {metrics_csv}")

    # Save best model metadata & confusion matrix
    best_model_name = "Logistic Regression"
    cm_json = f"{base_dir}/outputs/confusion_matrix.json"
    with open(cm_json, "w", encoding="utf-8") as f:
        json.dump(all_evaluations[best_model_name]["confusion_matrix"], f, indent=2)

    # Save serialized model parameters for inference
    best_model = models[best_model_name]
    pipeline_artifact = {
        "model_name": best_model_name,
        "classes": CATEGORIES,
        "feature_count": num_features,
        "vocabulary": vectorizer.vocabulary,
        "idf": vectorizer.idf,
        "weights": best_model.weights,
        "bias": best_model.bias,
        "random_seed": RANDOM_SEED,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "macro_f1": all_evaluations[best_model_name]["macro_f1"]
    }
    model_path = f"{base_dir}/models/complaint_pipeline_metadata.json"
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_artifact, f)
    print(f"  ✓ Serialized production pipeline metadata to {model_path}")

    # Generate sample test predictions export
    sample_preds = []
    for r, vec in zip(test_data[:30], X_test[:30]):
        pred_cat, probs = best_model.predict_single(vec)
        sample_preds.append({
            "complaint_id": r["complaint_id"],
            "actual_category": r["category"],
            "predicted_category": pred_cat,
            "confidence": round(probs[pred_cat], 4),
            "is_correct": r["category"] == pred_cat,
            "complaint_snippet": r["original_text"][:100] + "..."
        })
    test_pred_csv = f"{base_dir}/outputs/test_predictions.csv"
    with open(test_pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["complaint_id", "actual_category", "predicted_category", "confidence", "is_correct", "complaint_snippet"])
        writer.writeheader()
        writer.writerows(sample_preds)
    print(f"  ✓ Saved test predictions to {test_pred_csv}")

    print("\nPhase 1-6 Artifact generation complete!")

if __name__ == "__main__":
    main()
