// Comprehensive Hospital Database for India (State and District wise)

export interface Hospital {
  name: string;
  type: 'Government Hospital' | 'Private Hospital' | 'Nursing Home' | 'Medical College Hospital' | 'Primary Health Center';
  address: string;
  location: string; // e.g., "Jadavpur, Kolkata"
  state: string;
  district: string;
  phone: string;
  emergencyPhone: string;
  website: string;
  hours: string;
  services: string;
  category: 'govt' | 'private';
}

export const hospitalDatabase: Hospital[] = [
  // West Bengal - Kolkata
  {
    name: "SSKM Hospital (Kolkata Medical College)",
    type: "Medical College Hospital",
    address: "244 AJC Bose Road",
    location: "Bhowanipore, Kolkata",
    state: "West Bengal",
    district: "Kolkata",
    phone: "+91-033-2223-3636",
    emergencyPhone: "+91-033-2223-5000",
    website: "https://www.kmcgov.in/",
    hours: "24/7 Emergency Services",
    services: "TB care, MDR-TB treatment, DOTS therapy, X-ray, CT scan",
    category: "govt"
  },
  {
    name: "R.G. Kar Medical College & Hospital",
    type: "Medical College Hospital",
    address: "1, Khudiram Bose Sarani",
    location: "Belgachia, Kolkata",
    state: "West Bengal",
    district: "Kolkata",
    phone: "+91-033-2555-7656",
    emergencyPhone: "+91-033-2555-7777",
    website: "https://www.rgkarmch.in/",
    hours: "24/7 Emergency Services",
    services: "Comprehensive TB care, drug-resistant TB treatment, pulmonology",
    category: "govt"
  },
  {
    name: "Jadavpur TB Hospital",
    type: "Government Hospital",
    address: "Raja SC Mallick Road",
    location: "Jadavpur, Kolkata",
    state: "West Bengal",
    district: "Kolkata",
    phone: "+91-033-2414-0101",
    emergencyPhone: "+91-033-2414-0102",
    website: "https://www.wbhealth.gov.in/",
    hours: "Mon-Sat: 9:00 AM - 5:00 PM",
    services: "Free TB testing, DOTS therapy, X-ray facilities",
    category: "govt"
  },
  {
    name: "Apollo Gleneagles Hospitals",
    type: "Private Hospital",
    address: "58, Canal Circular Road",
    location: "Kadapara, Kolkata",
    state: "West Bengal",
    district: "Kolkata",
    phone: "+91-033-2320-3040",
    emergencyPhone: "+91-033-2320-2122",
    website: "https://www.apollogleneagles.in/",
    hours: "24/7 Services",
    services: "Advanced TB diagnostics, specialist consultation, ICU facilities",
    category: "private"
  },
  {
    name: "Fortis Hospital",
    type: "Private Hospital",
    address: "730, Anandapur, EM Bypass Road",
    location: "Anandapur, Kolkata",
    state: "West Bengal",
    district: "Kolkata",
    phone: "+91-033-6628-4444",
    emergencyPhone: "+91-033-6628-4000",
    website: "https://www.fortishealthcare.com/kolkata",
    hours: "24/7 Services",
    services: "TB treatment, pulmonology, radiology, ICU",
    category: "private"
  },
  {
    name: "Peerless Hospital",
    type: "Private Hospital",
    address: "360, Panchasayar",
    location: "Garia, Kolkata",
    state: "West Bengal",
    district: "Kolkata",
    phone: "+91-033-2441-5151",
    emergencyPhone: "+91-033-2441-1212",
    website: "https://www.peerlesshospital.com/",
    hours: "24/7 Services",
    services: "TB care, chest medicine, diagnostics",
    category: "private"
  },
  {
    name: "Salt Lake City Nursing Home",
    type: "Nursing Home",
    address: "Block A, Sector 1",
    location: "Salt Lake, Kolkata",
    state: "West Bengal",
    district: "Kolkata",
    phone: "+91-033-2337-4000",
    emergencyPhone: "+91-033-2337-4100",
    website: "https://www.saltlakenursinghome.com/",
    hours: "Mon-Sun: 8:00 AM - 10:00 PM",
    services: "TB screening, general consultation",
    category: "private"
  },

  // West Bengal - Murshidabad
  {
    name: "Murshidabad Medical College & Hospital",
    type: "Medical College Hospital",
    address: "Berhampore College Road",
    location: "Berhampore, Murshidabad",
    state: "West Bengal",
    district: "Murshidabad",
    phone: "+91-03482-270303",
    emergencyPhone: "+91-03482-270400",
    website: "https://www.mmch.gov.in/",
    hours: "24/7 Emergency Services",
    services: "Comprehensive TB care, DOTS therapy, MDR-TB treatment",
    category: "govt"
  },
  {
    name: "District TB Center Berhampore",
    type: "Government Hospital",
    address: "Civil Hospital Compound",
    location: "Berhampore, Murshidabad",
    state: "West Bengal",
    district: "Murshidabad",
    phone: "+91-03482-250123",
    emergencyPhone: "+91-03482-250200",
    website: "https://www.wbhealth.gov.in/",
    hours: "Mon-Sat: 9:00 AM - 5:00 PM",
    services: "Free TB testing, DOTS centers, X-ray",
    category: "govt"
  },
  {
    name: "Jiaganj Rural Hospital",
    type: "Government Hospital",
    address: "Jiaganj Main Road",
    location: "Jiaganj, Murshidabad",
    state: "West Bengal",
    district: "Murshidabad",
    phone: "+91-03482-262345",
    emergencyPhone: "+91-03482-262400",
    website: "https://www.wbhealth.gov.in/",
    hours: "Mon-Sat: 8:00 AM - 6:00 PM",
    services: "TB screening, medication distribution",
    category: "govt"
  },

  // West Bengal - North 24 Parganas
  {
    name: "Barasat District Hospital",
    type: "Government Hospital",
    address: "Barasat Station Road",
    location: "Barasat, North 24 Parganas",
    state: "West Bengal",
    district: "North 24 Parganas",
    phone: "+91-033-2543-2100",
    emergencyPhone: "+91-033-2543-2200",
    website: "https://www.wbhealth.gov.in/",
    hours: "24/7 Emergency Services",
    services: "TB care, DOTS therapy, emergency services",
    category: "govt"
  },
  {
    name: "ILS Hospital",
    type: "Private Hospital",
    address: "DD-6, Sector 1",
    location: "Salt Lake, North 24 Parganas",
    state: "West Bengal",
    district: "North 24 Parganas",
    phone: "+91-033-4040-8080",
    emergencyPhone: "+91-033-4040-8000",
    website: "https://www.ilshospitals.com/",
    hours: "24/7 Services",
    services: "TB diagnostics, pulmonology, ICU",
    category: "private"
  },

  // West Bengal - Howrah
  {
    name: "Howrah General Hospital",
    type: "Government Hospital",
    address: "College Street",
    location: "Howrah, Howrah",
    state: "West Bengal",
    district: "Howrah",
    phone: "+91-033-2660-4141",
    emergencyPhone: "+91-033-2660-4000",
    website: "https://www.wbhealth.gov.in/",
    hours: "24/7 Emergency Services",
    services: "Free TB testing, DOTS centers",
    category: "govt"
  },

  // Maharashtra - Mumbai
  {
    name: "KEM Hospital",
    type: "Medical College Hospital",
    address: "489, Sardar Moodliar Road, Parel",
    location: "Parel, Mumbai",
    state: "Maharashtra",
    district: "Mumbai",
    phone: "+91-022-2410-7000",
    emergencyPhone: "+91-022-2410-7777",
    website: "https://www.kem.edu/",
    hours: "24/7 Emergency Services",
    services: "Comprehensive TB care, MDR-TB treatment, research",
    category: "govt"
  },
  {
    name: "Sion Hospital",
    type: "Government Hospital",
    address: "Sion-Trombay Road",
    location: "Sion, Mumbai",
    state: "Maharashtra",
    district: "Mumbai",
    phone: "+91-022-2407-6101",
    emergencyPhone: "+91-022-2407-6000",
    website: "https://www.sionhospital.org/",
    hours: "24/7 Emergency Services",
    services: "TB care, DOTS therapy, chest clinic",
    category: "govt"
  },
  {
    name: "Lilavati Hospital",
    type: "Private Hospital",
    address: "A-791, Bandra Reclamation",
    location: "Bandra, Mumbai",
    state: "Maharashtra",
    district: "Mumbai",
    phone: "+91-022-2640-0000",
    emergencyPhone: "+91-022-2640-0888",
    website: "https://www.lilavatihospital.com/",
    hours: "24/7 Services",
    services: "Advanced TB diagnostics, pulmonology",
    category: "private"
  },
  {
    name: "Hinduja Hospital",
    type: "Private Hospital",
    address: "Veer Savarkar Marg, Mahim",
    location: "Mahim, Mumbai",
    state: "Maharashtra",
    district: "Mumbai",
    phone: "+91-022-2444-9199",
    emergencyPhone: "+91-022-2444-1515",
    website: "https://www.hindujahospital.com/",
    hours: "24/7 Services",
    services: "TB treatment, specialist consultation",
    category: "private"
  },
  {
    name: "Nanavati Hospital",
    type: "Private Hospital",
    address: "SV Road, Vile Parle West",
    location: "Vile Parle, Mumbai",
    state: "Maharashtra",
    district: "Mumbai",
    phone: "+91-022-2626-7777",
    emergencyPhone: "+91-022-2626-7888",
    website: "https://www.nanavatihospital.org/",
    hours: "24/7 Services",
    services: "Comprehensive TB care, radiology",
    category: "private"
  },

  // Maharashtra - Pune
  {
    name: "Sassoon General Hospital",
    type: "Government Hospital",
    address: "Near Pune Railway Station",
    location: "Pune, Pune",
    state: "Maharashtra",
    district: "Pune",
    phone: "+91-020-2612-1071",
    emergencyPhone: "+91-020-2612-1000",
    website: "https://www.sassoonhospitals.in/",
    hours: "24/7 Emergency Services",
    services: "Free TB care, DOTS therapy, emergency",
    category: "govt"
  },
  {
    name: "Ruby Hall Clinic",
    type: "Private Hospital",
    address: "40, Sassoon Road",
    location: "Pune, Pune",
    state: "Maharashtra",
    district: "Pune",
    phone: "+91-020-6645-8888",
    emergencyPhone: "+91-020-6645-8000",
    website: "https://www.rubyhall.com/",
    hours: "24/7 Services",
    services: "TB diagnostics, pulmonology department",
    category: "private"
  },
  {
    name: "Deenanath Mangeshkar Hospital",
    type: "Private Hospital",
    address: "Erandwane",
    location: "Erandwane, Pune",
    state: "Maharashtra",
    district: "Pune",
    phone: "+91-020-6710-3000",
    emergencyPhone: "+91-020-6710-3333",
    website: "https://www.dmhospital.org/",
    hours: "24/7 Services",
    services: "TB treatment, chest medicine",
    category: "private"
  },

  // Delhi - Central Delhi
  {
    name: "Lady Hardinge Medical College",
    type: "Medical College Hospital",
    address: "Connaught Place",
    location: "Connaught Place, Central Delhi",
    state: "Delhi",
    district: "Central Delhi",
    phone: "+91-011-2336-5525",
    emergencyPhone: "+91-011-2336-5000",
    website: "https://www.lhmc-du.ac.in/",
    hours: "24/7 Emergency Services",
    services: "Comprehensive TB care, MDR-TB treatment",
    category: "govt"
  },
  {
    name: "Ram Manohar Lohia Hospital",
    type: "Government Hospital",
    address: "Baba Kharak Singh Marg",
    location: "Connaught Place, Central Delhi",
    state: "Delhi",
    district: "Central Delhi",
    phone: "+91-011-2336-5525",
    emergencyPhone: "+91-011-2336-5000",
    website: "https://www.rmlh.nic.in/",
    hours: "24/7 Emergency Services",
    services: "Free TB testing, DOTS therapy",
    category: "govt"
  },

  // Delhi - South Delhi
  {
    name: "AIIMS Delhi",
    type: "Medical College Hospital",
    address: "Ansari Nagar",
    location: "Ansari Nagar, South Delhi",
    state: "Delhi",
    district: "South Delhi",
    phone: "+91-011-2658-8500",
    emergencyPhone: "+91-011-2658-8700",
    website: "https://www.aiims.edu/",
    hours: "24/7 Emergency Services",
    services: "Advanced TB care, research, MDR-TB treatment",
    category: "govt"
  },
  {
    name: "Safdarjung Hospital",
    type: "Government Hospital",
    address: "Ansari Nagar West",
    location: "Safdarjung, South Delhi",
    state: "Delhi",
    district: "South Delhi",
    phone: "+91-011-2616-5060",
    emergencyPhone: "+91-011-2616-5000",
    website: "https://www.vmmc-sjh.nic.in/",
    hours: "24/7 Emergency Services",
    services: "TB care, chest clinic, DOTS",
    category: "govt"
  },
  {
    name: "Max Super Speciality Hospital",
    type: "Private Hospital",
    address: "Press Enclave Road",
    location: "Saket, South Delhi",
    state: "Delhi",
    district: "South Delhi",
    phone: "+91-011-2651-5050",
    emergencyPhone: "+91-011-2651-5000",
    website: "https://www.maxhealthcare.in/",
    hours: "24/7 Services",
    services: "Multi-specialty TB care, pulmonology",
    category: "private"
  },
  {
    name: "Fortis Escorts Heart Institute",
    type: "Private Hospital",
    address: "Okhla Road",
    location: "Okhla, South Delhi",
    state: "Delhi",
    district: "South Delhi",
    phone: "+91-011-4713-5000",
    emergencyPhone: "+91-011-4713-5555",
    website: "https://www.fortishealthcare.com/delhi",
    hours: "24/7 Services",
    services: "TB treatment, chest medicine, ICU",
    category: "private"
  },

  // Karnataka - Bangalore
  {
    name: "Victoria Hospital (BMCRI)",
    type: "Medical College Hospital",
    address: "Fort, K.R. Road",
    location: "Fort, Bangalore",
    state: "Karnataka",
    district: "Bangalore Urban",
    phone: "+91-080-2670-1150",
    emergencyPhone: "+91-080-2670-1000",
    website: "https://www.bmcri.edu.in/",
    hours: "24/7 Emergency Services",
    services: "Comprehensive TB care, DOTS therapy, MDR-TB",
    category: "govt"
  },
  {
    name: "Rajiv Gandhi Institute of Chest Diseases",
    type: "Government Hospital",
    address: "Magadi Road",
    location: "Rajajinagar, Bangalore",
    state: "Karnataka",
    district: "Bangalore Urban",
    phone: "+91-080-2331-5678",
    emergencyPhone: "+91-080-2331-5000",
    website: "https://www.rgicd.karnataka.gov.in/",
    hours: "Mon-Sat: 9:00 AM - 5:00 PM",
    services: "Specialized TB treatment, drug-resistant TB",
    category: "govt"
  },
  {
    name: "Manipal Hospital",
    type: "Private Hospital",
    address: "98, HAL Airport Road",
    location: "Old Airport Road, Bangalore",
    state: "Karnataka",
    district: "Bangalore Urban",
    phone: "+91-080-2502-4444",
    emergencyPhone: "+91-080-2502-4000",
    website: "https://www.manipalhospitals.com/",
    hours: "24/7 Services",
    services: "TB diagnostics, pulmonology, specialist care",
    category: "private"
  },
  {
    name: "Apollo Hospitals",
    type: "Private Hospital",
    address: "154/11, Bannerghatta Road",
    location: "Bannerghatta, Bangalore",
    state: "Karnataka",
    district: "Bangalore Urban",
    phone: "+91-080-2630-2244",
    emergencyPhone: "+91-080-2630-2000",
    website: "https://www.apollohospitals.com/bangalore",
    hours: "24/7 Services",
    services: "Advanced TB care, radiology, ICU",
    category: "private"
  },

  // Tamil Nadu - Chennai
  {
    name: "Government Stanley Medical College",
    type: "Medical College Hospital",
    address: "Old Jail Road",
    location: "Royapuram, Chennai",
    state: "Tamil Nadu",
    district: "Chennai",
    phone: "+91-044-2528-2402",
    emergencyPhone: "+91-044-2528-2000",
    website: "https://www.tnmgrmu.ac.in/",
    hours: "24/7 Emergency Services",
    services: "Comprehensive TB care, free treatment, DOTS",
    category: "govt"
  },
  {
    name: "Government Hospital of Thoracic Medicine",
    type: "Government Hospital",
    address: "Tambaram Sanatorium",
    location: "Tambaram, Chennai",
    state: "Tamil Nadu",
    district: "Chennai",
    phone: "+91-044-2223-4567",
    emergencyPhone: "+91-044-2223-4000",
    website: "https://www.tnhealth.org/",
    hours: "24/7 Emergency Services",
    services: "Specialized TB treatment, MDR-TB care",
    category: "govt"
  },
  {
    name: "Apollo Hospitals",
    type: "Private Hospital",
    address: "21, Greams Lane",
    location: "Greams Road, Chennai",
    state: "Tamil Nadu",
    district: "Chennai",
    phone: "+91-044-2829-0200",
    emergencyPhone: "+91-044-2829-0000",
    website: "https://www.apollohospitals.com/chennai",
    hours: "24/7 Services",
    services: "Advanced TB diagnostics, pulmonology",
    category: "private"
  },
  {
    name: "Fortis Malar Hospital",
    type: "Private Hospital",
    address: "52, 1st Main Road, Gandhi Nagar",
    location: "Adyar, Chennai",
    state: "Tamil Nadu",
    district: "Chennai",
    phone: "+91-044-4289-2222",
    emergencyPhone: "+91-044-4289-2000",
    website: "https://www.fortishealthcare.com/chennai",
    hours: "24/7 Services",
    services: "TB treatment, chest medicine",
    category: "private"
  },

  // Uttar Pradesh - Lucknow
  {
    name: "King George's Medical University",
    type: "Medical College Hospital",
    address: "Shah Mina Road",
    location: "Chowk, Lucknow",
    state: "Uttar Pradesh",
    district: "Lucknow",
    phone: "+91-0522-220-7777",
    emergencyPhone: "+91-0522-220-7000",
    website: "https://www.kgmu.org/",
    hours: "24/7 Emergency Services",
    services: "Comprehensive TB care, MDR-TB treatment, research",
    category: "govt"
  },
  {
    name: "Vallabhbhai Patel Chest Institute",
    type: "Government Hospital",
    address: "Aliganj",
    location: "Aliganj, Lucknow",
    state: "Uttar Pradesh",
    district: "Lucknow",
    phone: "+91-0522-233-4567",
    emergencyPhone: "+91-0522-233-4000",
    website: "https://www.uphealth.gov.in/",
    hours: "Mon-Sat: 9:00 AM - 5:00 PM",
    services: "Specialized TB treatment, DOTS therapy",
    category: "govt"
  },
  {
    name: "Sahara Hospital",
    type: "Private Hospital",
    address: "Viraj Khand, Gomti Nagar",
    location: "Gomti Nagar, Lucknow",
    state: "Uttar Pradesh",
    district: "Lucknow",
    phone: "+91-0522-398-8888",
    emergencyPhone: "+91-0522-398-8000",
    website: "https://www.saharahospital.com/",
    hours: "24/7 Services",
    services: "TB diagnostics, pulmonology department",
    category: "private"
  },

  // Gujarat - Ahmedabad
  {
    name: "Civil Hospital Ahmedabad",
    type: "Government Hospital",
    address: "Asarwa",
    location: "Asarwa, Ahmedabad",
    state: "Gujarat",
    district: "Ahmedabad",
    phone: "+91-079-2268-9011",
    emergencyPhone: "+91-079-2268-9000",
    website: "https://www.nhmgujarat.gov.in/",
    hours: "24/7 Emergency Services",
    services: "Free TB testing, DOTS centers, emergency care",
    category: "govt"
  },
  {
    name: "Sterling Hospital",
    type: "Private Hospital",
    address: "Off Gurukul Road",
    location: "Memnagar, Ahmedabad",
    state: "Gujarat",
    district: "Ahmedabad",
    phone: "+91-079-6777-7777",
    emergencyPhone: "+91-079-6777-7000",
    website: "https://www.sterlinghospitals.com/",
    hours: "24/7 Services",
    services: "TB treatment, pulmonology, ICU",
    category: "private"
  },
  {
    name: "Apollo Hospitals",
    type: "Private Hospital",
    address: "Plot No. 1A, GIDC Estate, Bhat",
    location: "Gandhinagar, Ahmedabad",
    state: "Gujarat",
    district: "Ahmedabad",
    phone: "+91-079-6670-1234",
    emergencyPhone: "+91-079-6670-1000",
    website: "https://www.apollohospitals.com/ahmedabad",
    hours: "24/7 Services",
    services: "Advanced TB diagnostics, specialist consultation",
    category: "private"
  },
];

// Filter hospitals by state
export const getHospitalsByState = (state: string): Hospital[] => {
  return hospitalDatabase.filter(h => h.state === state);
};

// Filter hospitals by state and district
export const getHospitalsByDistrict = (state: string, district: string): Hospital[] => {
  return hospitalDatabase.filter(h => h.state === state && h.district === district);
};

// Filter hospitals by state, district, and city
export const getHospitalsByCity = (state: string, district: string, city: string): Hospital[] => {
  return hospitalDatabase.filter(h => 
    h.state === state && 
    h.district === district && 
    h.location.toLowerCase().includes(city.toLowerCase())
  );
};

// Search hospitals by name
export const searchHospitalsByName = (query: string): Hospital[] => {
  const normalizedQuery = query.toLowerCase().trim();
  return hospitalDatabase.filter(h => 
    h.name.toLowerCase().includes(normalizedQuery) ||
    h.location.toLowerCase().includes(normalizedQuery)
  );
};
