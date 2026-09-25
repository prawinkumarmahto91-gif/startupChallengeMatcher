"""
generate_dataset.py

Creates synthetic startups.csv and challenges.csv for the
Startup-Challenge Matching project.

This is a one-time setup script (Day 2). Run it once to create the data,
then move on to Day 3. If you want more data later, just increase
TOTAL_STARTUPS / TOTAL_CHALLENGES below and rerun.
"""

import random
import csv
import os

# ---------------------------------------------------------------------------
# CONFIG - change these numbers later to scale up the dataset (e.g. 100, 20)
# ---------------------------------------------------------------------------
TOTAL_STARTUPS = 40
TOTAL_CHALLENGES = 15
RANDOM_SEED = 42  # keeps the "random" data identical every time we run this

random.seed(RANDOM_SEED)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Reference vocabulary used to build realistic synthetic records
# ---------------------------------------------------------------------------

SECTOR_TECH_POOL = {
    "Waste Management": ["AI route optimization", "IoT sensors", "GPS tracking",
                          "Computer vision", "Waste sorting robotics"],
    "Healthcare": ["AI diagnostics", "Predictive analytics", "IoT patient monitoring",
                   "Queue management software", "Telemedicine"],
    "Agriculture": ["Computer vision", "Drone imaging", "Satellite imagery",
                     "IoT soil sensors", "ML disease detection"],
    "Smart City": ["Computer vision", "AI traffic prediction", "IoT sensors",
                    "Adaptive signal control", "GPS tracking"],
    "Fintech": ["Fraud detection ML", "Blockchain", "Risk scoring models",
                "Credit analytics", "Real-time transaction monitoring"],
    "Water Management": ["IoT leak detection", "Acoustic sensors", "Satellite imagery",
                          "Predictive maintenance ML", "SCADA integration"],
    "Public Safety": ["Computer vision surveillance", "Predictive policing analytics",
                       "IoT emergency sensors", "GPS tracking"],
    "Education": ["Adaptive learning AI", "Learning analytics", "NLP tutoring",
                  "Attendance tracking IoT", "Content recommendation ML"],
    "Energy": ["Smart grid analytics", "IoT energy meters", "Predictive maintenance ML",
               "Demand forecasting AI", "Solar output prediction"],
}

CITIES = ["Pune", "Mumbai", "Bengaluru", "Delhi", "Chennai", "Hyderabad",
          "Ahmedabad", "Indore", "Jaipur", "Lucknow", "Bhopal", "Nagpur",
          "Kochi", "Surat", "Patna"]

CAPABILITIES_POOL = [
    "Real-time data dashboards", "Predictive analytics", "API integration with existing govt systems",
    "Field agent mobile app", "Scalable cloud infrastructure", "Multi-language support",
    "24/7 monitoring", "Custom reporting",
]

PREV_PROJECT_POOL = [
    "Deployed with Pune Municipal Corporation",
    "Piloted with Chennai Smart City Mission",
    "Completed pilot with State Health Department",
    "Worked with private sector clients only, no government experience",
    "Deployed with Bengaluru Traffic Police",
    "No prior deployment experience",
    "Piloted with a district administration office",
]

CERT_POOL = ["ISO 9001", "ISO 27001", "CMMI Level 3", "GDPR Compliant", "None"]

NAME_PREFIX = ["Eco", "Smart", "Urban", "Agri", "Health", "Fin", "Water", "Safe",
               "Grid", "Edu", "Civic", "Bright", "Next", "Clear", "Prime"]
NAME_MID = ["Route", "Vision", "Sense", "Watch", "Guard", "Sync", "Flow",
            "Track", "Nova", "Pulse", "Link", "Core", "Wave", "Shield"]
NAME_SUFFIX = ["AI", "Tech", "Labs", "Systems", "Solutions", "Technologies", "Analytics"]


def random_tech_list(sector: str, k: int = 3) -> str:
    """Pick k technologies relevant to a sector and join as a comma string."""
    pool = SECTOR_TECH_POOL[sector]
    k = min(k, len(pool))
    return ", ".join(random.sample(pool, k))


def random_capabilities(k: int = 3) -> str:
    k = min(k, len(CAPABILITIES_POOL))
    return ", ".join(random.sample(CAPABILITIES_POOL, k))


def random_certifications() -> str:
    # Small chance of having no certification at all
    choice = random.choice(CERT_POOL + CERT_POOL + ["None"])
    return choice


# ---------------------------------------------------------------------------
# STARTUPS - hand-written realistic entries from the project spec
# ---------------------------------------------------------------------------

named_startups = [
    {
        "startup_id": 1, "name": "EcoRoute AI",
        "description": "AI-powered route optimization for municipal waste collection fleets.",
        "sector": "Waste Management",
        "technologies": "AI route optimization, GPS tracking, IoT sensors",
        "capabilities": "Real-time data dashboards, API integration with existing govt systems",
        "experience": 4, "previous_projects": "Deployed with Pune Municipal Corporation",
        "location": "Pune", "budget": 45, "certifications": "ISO 9001", "dpiit_recognized": True,
    },
    {
        "startup_id": 2, "name": "HealthQueue Technologies",
        "description": "Patient flow and queue management software for government hospitals.",
        "sector": "Healthcare",
        "technologies": "Queue management software, Predictive analytics, IoT patient monitoring",
        "capabilities": "Real-time data dashboards, 24/7 monitoring",
        "experience": 6, "previous_projects": "Completed pilot with State Health Department",
        "location": "Hyderabad", "budget": 60, "certifications": "ISO 27001", "dpiit_recognized": True,
    },
    {
        "startup_id": 3, "name": "AgriVision Labs",
        "description": "Computer vision system for early detection of crop diseases from field images.",
        "sector": "Agriculture",
        "technologies": "Computer vision, ML disease detection, Drone imaging",
        "capabilities": "Field agent mobile app, Custom reporting",
        "experience": 3, "previous_projects": "Piloted with a district administration office",
        "location": "Indore", "budget": 30, "certifications": "None", "dpiit_recognized": True,
    },
    {
        "startup_id": 4, "name": "CleanCity IoT",
        "description": "IoT-based smart bins and sensor network for waste collection monitoring.",
        "sector": "Waste Management",
        "technologies": "IoT sensors, GPS tracking, Computer vision",
        "capabilities": "Real-time data dashboards, Scalable cloud infrastructure",
        "experience": 5, "previous_projects": "Deployed with Pune Municipal Corporation",
        "location": "Mumbai", "budget": 55, "certifications": "ISO 9001", "dpiit_recognized": True,
    },
    {
        "startup_id": 5, "name": "FinGuard AI",
        "description": "Fraud detection and risk scoring for government welfare disbursement systems.",
        "sector": "Fintech",
        "technologies": "Fraud detection ML, Risk scoring models, Real-time transaction monitoring",
        "capabilities": "API integration with existing govt systems, 24/7 monitoring",
        "experience": 7, "previous_projects": "Worked with private sector clients only, no government experience",
        "location": "Bengaluru", "budget": 80, "certifications": "ISO 27001", "dpiit_recognized": False,
    },
    {
        "startup_id": 6, "name": "UrbanTraffic AI",
        "description": "AI-based adaptive traffic signal control and congestion prediction.",
        "sector": "Smart City",
        "technologies": "AI traffic prediction, Adaptive signal control, Computer vision",
        "capabilities": "Real-time data dashboards, Scalable cloud infrastructure",
        "experience": 5, "previous_projects": "Deployed with Bengaluru Traffic Police",
        "location": "Bengaluru", "budget": 70, "certifications": "ISO 9001", "dpiit_recognized": True,
    },
    {
        "startup_id": 7, "name": "MedAssist Systems",
        "description": "AI diagnostic support and patient triage system for public hospitals.",
        "sector": "Healthcare",
        "technologies": "AI diagnostics, Predictive analytics, Telemedicine",
        "capabilities": "Multi-language support, Custom reporting",
        "experience": 2, "previous_projects": "No prior deployment experience",
        "location": "Delhi", "budget": 40, "certifications": "None", "dpiit_recognized": False,
    },
    {
        "startup_id": 8, "name": "WaterWatch Technologies",
        "description": "Acoustic and IoT sensor network for detecting underground water pipe leaks.",
        "sector": "Water Management",
        "technologies": "IoT leak detection, Acoustic sensors, Predictive maintenance ML",
        "capabilities": "Real-time data dashboards, API integration with existing govt systems",
        "experience": 6, "previous_projects": "Piloted with Chennai Smart City Mission",
        "location": "Chennai", "budget": 50, "certifications": "ISO 9001", "dpiit_recognized": True,
    },
    {
        "startup_id": 9, "name": "RouteSmart Solutions",
        "description": "Fleet and route optimization platform adaptable to waste and logistics use cases.",
        "sector": "Waste Management",
        "technologies": "AI route optimization, GPS tracking",
        "capabilities": "Real-time data dashboards, Field agent mobile app",
        "experience": 3, "previous_projects": "No prior deployment experience",
        "location": "Jaipur", "budget": 35, "certifications": "None", "dpiit_recognized": True,
    },
]

# ---------------------------------------------------------------------------
# CHALLENGES - hand-written realistic entries from the project spec
# ---------------------------------------------------------------------------

named_challenges = [
    {
        "challenge_id": 101, "title": "Waste Collection Optimization",
        "problem": "Municipal waste trucks follow fixed routes regardless of actual bin fill "
                   "levels, wasting fuel and missing overflowing bins.",
        "sector": "Waste Management",
        "technologies": "AI route optimization, GPS tracking, IoT sensors",
        "desired_outcome": "Reduce fuel cost and missed collections through dynamic routing.",
        "location": "Pune", "budget": 50,
        "mandatory_requirements": "DPIIT recognition required",
    },
    {
        "challenge_id": 102, "title": "Hospital Patient Flow Optimization",
        "problem": "Government hospitals face long patient queues and inefficient bed allocation.",
        "sector": "Healthcare",
        "technologies": "Queue management software, Predictive analytics",
        "desired_outcome": "Reduce average patient wait time and improve bed utilization.",
        "location": "Hyderabad", "budget": 65,
        "mandatory_requirements": "ISO 27001 certification required",
    },
    {
        "challenge_id": 103, "title": "Crop Disease Detection",
        "problem": "Farmers lack early warning tools for crop disease, leading to yield loss.",
        "sector": "Agriculture",
        "technologies": "Computer vision, ML disease detection",
        "desired_outcome": "Early detection of crop disease via field-image analysis.",
        "location": "Indore", "budget": 30,
        "mandatory_requirements": "No mandatory certification",
    },
    {
        "challenge_id": 104, "title": "Urban Traffic Management",
        "problem": "Fixed-timing traffic signals cause congestion during peak hours in growing cities.",
        "sector": "Smart City",
        "technologies": "AI traffic prediction, Adaptive signal control",
        "desired_outcome": "Reduce average commute time through adaptive signal control.",
        "location": "Bengaluru", "budget": 75,
        "mandatory_requirements": "DPIIT recognition required",
    },
    {
        "challenge_id": 105, "title": "Water Leak Detection",
        "problem": "Underground pipeline leaks go undetected for months, wasting treated water.",
        "sector": "Water Management",
        "technologies": "IoT leak detection, Acoustic sensors",
        "desired_outcome": "Detect and localize leaks within days instead of months.",
        "location": "Chennai", "budget": 45,
        "mandatory_requirements": "No mandatory certification",
    },
]

# ---------------------------------------------------------------------------
# Auto-generate the remaining rows to reach TOTAL_STARTUPS / TOTAL_CHALLENGES
# ---------------------------------------------------------------------------

def generate_extra_startups(existing: list, target_total: int) -> list:
    used_names = {s["name"] for s in existing}
    sectors = list(SECTOR_TECH_POOL.keys())
    next_id = max(s["startup_id"] for s in existing) + 1
    extra = []

    while len(existing) + len(extra) < target_total:
        sector = random.choice(sectors)
        name = f"{random.choice(NAME_PREFIX)}{random.choice(NAME_MID)} {random.choice(NAME_SUFFIX)}"
        if name in used_names:
            continue
        used_names.add(name)

        row = {
            "startup_id": next_id,
            "name": name,
            "description": f"{name} provides {sector.lower()} technology solutions to help "
                            f"government bodies improve operational efficiency.",
            "sector": sector,
            "technologies": random_tech_list(sector, k=random.choice([2, 3])),
            "capabilities": random_capabilities(k=random.choice([2, 3])),
            "experience": random.randint(1, 10),
            "previous_projects": random.choice(PREV_PROJECT_POOL),
            "location": random.choice(CITIES),
            "budget": random.randint(15, 150),
            "certifications": random_certifications(),
            "dpiit_recognized": random.random() < 0.65,
        }
        extra.append(row)
        next_id += 1

    return existing + extra


CHALLENGE_TITLE_POOL = {
    "Public Safety": "Predictive Policing for Crime Hotspots",
    "Education": "Digital Attendance and Learning Tracking",
    "Energy": "Smart Grid Load Balancing",
    "Fintech": "Fraud Detection in Welfare Disbursement",
    "Smart City": "Smart Parking Management",
}

REQUIREMENT_POOL = [
    "DPIIT recognition required",
    "ISO 27001 certification required",
    "No mandatory certification",
    "Minimum 3 years of experience required",
]


def generate_extra_challenges(existing: list, target_total: int) -> list:
    sectors = list(SECTOR_TECH_POOL.keys())
    next_id = max(c["challenge_id"] for c in existing) + 1
    extra = []
    used_titles = {c["title"] for c in existing}

    while len(existing) + len(extra) < target_total:
        sector = random.choice(sectors)
        title = CHALLENGE_TITLE_POOL.get(sector, f"{sector} Improvement Initiative")
        if title in used_titles:
            title = f"{title} - Phase {random.randint(2, 5)}"
        used_titles.add(title)

        row = {
            "challenge_id": next_id,
            "title": title,
            "problem": f"The government department is facing operational inefficiencies in "
                       f"{sector.lower()} that require a technology-driven solution.",
            "sector": sector,
            "technologies": random_tech_list(sector, k=random.choice([2, 3])),
            "desired_outcome": f"Improve measurable outcomes related to {sector.lower()} operations.",
            "location": random.choice(CITIES),
            "budget": random.randint(20, 100),
            "mandatory_requirements": random.choice(REQUIREMENT_POOL),
        }
        extra.append(row)
        next_id += 1

    return existing + extra


all_startups = generate_extra_startups(named_startups, TOTAL_STARTUPS)
all_challenges = generate_extra_challenges(named_challenges, TOTAL_CHALLENGES)

# ---------------------------------------------------------------------------
# Write to CSV
# ---------------------------------------------------------------------------

startup_fields = ["startup_id", "name", "description", "sector", "technologies",
                   "capabilities", "experience", "previous_projects", "location",
                   "budget", "certifications", "dpiit_recognized"]

challenge_fields = ["challenge_id", "title", "problem", "sector", "technologies",
                     "desired_outcome", "location", "budget", "mandatory_requirements"]


def write_csv(path: str, fieldnames: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


write_csv(os.path.join(DATA_DIR, "startups.csv"), startup_fields, all_startups)
write_csv(os.path.join(DATA_DIR, "challenges.csv"), challenge_fields, all_challenges)

print(f"Created {len(all_startups)} startups -> {DATA_DIR}/startups.csv")
print(f"Created {len(all_challenges)} challenges -> {DATA_DIR}/challenges.csv")
print("Done.")