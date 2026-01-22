// Expanded Hospital Database for India (Based on Wikipedia List of Hospitals in India)

export interface Hospital {
  name: string;
  type: 'Government Hospital' | 'Private Hospital' | 'Medical College Hospital' | 'District Hospital' | 'Community Health Center' | 'Primary Health Center' | 'Specialty Hospital';
  address: string;
  state: string;
  district: string;
  phone: string;
  emergencyPhone: string;
  website: string;
  hours: string;
  services: string;
  category: 'govt' | 'private';
}

export const expandedHospitalDatabase: Hospital[] = [
  // Andhra Pradesh - Visakhapatnam
  {
    name: "King George Hospital (KGH)",
    type: "Government Hospital",
    address: "Maharanipeta",
    state: "Andhra Pradesh",
    district: "Visakhapatnam",
    phone: "+91-0891-2564290",
    emergencyPhone: "+91-0891-2564291",
    website: "https://kghvisakhapatnam.ap.gov.in/",
    hours: "24/7 Emergency Services",
    services: "TB care, MDR-TB treatment, DOTS therapy, chest X-ray, CT scan",
    category: "govt"
  },
  {
    name: "Andhra Medical College",
    type: "Medical College Hospital",
    address: "Old Jail Road",
    state: "Andhra Pradesh",
    district: "Visakhapatnam",
    phone: "+91-0891-2555555",
    emergencyPhone: "+91-0891-2555000",
    website: "https://www.amcollege.edu.in/",
    hours: "24/7 Emergency Services",
    services: "Comprehensive TB care, drug-resistant TB treatment, pulmonology",
    category: "govt"
  },
  {
    name: "Seven Hills Hospital",
    type: "Private Hospital",
    address: "Rockdale Layout, Visakhapatnam",
    state: "Andhra Pradesh",
    district: "Visakhapatnam",
    phone: "+91-0891-6671919",
    emergencyPhone: "+91-0891-6671000",
    website: "https://www.7hillshealthcity.com/",
    hours: "24/7 Services",
    services: "Advanced TB diagnostics, pulmonology, ICU facilities",
    category: "private"
  },

  // Andhra Pradesh - Guntur
  {
    name: "Government General Hospital Guntur",
    type: "Government Hospital",
    address: "Kothapet, Guntur",
    state: "Andhra Pradesh",
    district: "Guntur",
    phone: "+91-0863-2222111",
    emergencyPhone: "+91-0863-2222000",
    website: "https://www.aphealth.gov.in/",
    hours: "24/7 Emergency Services",
    services: "Free TB testing, DOTS centers, emergency care",
    category: "govt"
  },
  {
    name: "Guntur Medical College",
    type: "Medical College Hospital",
    address: "Guntur",
    state: "Andhra Pradesh",
    district: "Guntur",
    phone: "+91-0863-2234567",
    emergencyPhone: "+91-0863-2234000",
    website: "https://www.gmcguntur.ac.in/",
    hours: "24/7 Emergency Services",
    services: "TB care, MDR-TB treatment, research facilities",
    category: "govt"
  },

  // Assam - Kamrup Metro (Guwahati)
  {
    name: "Gauhati Medical College Hospital (GMCH)",
    type: "Medical College Hospital",
    address: "Bhangagarh, Guwahati",
    state: "Assam",
    district: "Kamrup Metro",
    phone: "+91-0361-2528016",
    emergencyPhone: "+91-0361-2528000",
    website: "https://www.gmch.gov.in/",
    hours: "24/7 Emergency Services",
    services: "Comprehensive TB care, DOTS therapy, MDR-TB treatment",
    category: "govt"
  },
  {
    name: "Down Town Hospital",
    type: "Private Hospital",
    address: "GS Road, Guwahati",
    state: "Assam",
    district: "Kamrup Metro",
    phone: "+91-0361-2331003",
    emergencyPhone: "+91-0361-2331000",
    website: "https://www.downtownhospital.in/",
    hours: "24/7 Services",
    services: "TB diagnostics, pulmonology department, specialist care",
    category: "private"
  },
  {
    name: "GNRC Hospital",
    type: "Private Hospital",
    address: "Dispur, Guwahati",
    state: "Assam",
    district: "Kamrup Metro",
    phone: "+91-0361-2262242",
    emergencyPhone: "+91-0361-2262000",
    website: "https://www.gnrchospitals.com/",
    hours: "24/7 Services",
    services: "Advanced TB care, radiology, ICU",
    category: "private"
  },

  // Assam - Dibrugarh
  {
    name: "Assam Medical College Hospital",
    type: "Medical College Hospital",
    address: "Barbari, Dibrugarh",
    state: "Assam",
    district: "Dibrugarh",
    phone: "+91-0373-2301542",
    emergencyPhone: "+91-0373-2301000",
    website: "https://www.amcdibrugarh.ac.in/",
    hours: "24/7 Emergency Services",
    services: "TB care, chest medicine, DOTS therapy",
    category: "govt"
  },

  // Bihar - Patna
  {
    name: "Patna Medical College Hospital (PMCH)",
    type: "Medical College Hospital",
    address: "Ashok Rajpath, Patna",
    state: "Bihar",
    district: "Patna",
    phone: "+91-0612-2300025",
    emergencyPhone: "+91-0612-2300000",
    website: "https://www.pmch.edu.in/",
    hours: "24/7 Emergency Services",
    services: "Comprehensive TB care, free treatment, DOTS",
    category: "govt"
  },
  {
    name: "Indira Gandhi Institute of Medical Sciences (IGIMS)",
    type: "Medical College Hospital",
    address: "Sheikhpura, Patna",
    state: "Bihar",
    district: "Patna",
    phone: "+91-0612-2297382",
    emergencyPhone: "+91-0612-2297000",
    website: "https://www.igims.org/",
    hours: "24/7 Emergency Services",
    services: "TB treatment, MDR-TB care, pulmonology",
    category: "govt"
  },
  {
    name: "Paras HMRI Hospital",
    type: "Private Hospital",
    address: "NH-30, Bailey Road, Patna",
    state: "Bihar",
    district: "Patna",
    phone: "+91-0612-3540100",
    emergencyPhone: "+91-0612-3540000",
    website: "https://www.parashmrihospital.com/",
    hours: "24/7 Services",
    services: "Advanced TB diagnostics, specialist consultation",
    category: "private"
  },

  // Bihar - Gaya
  {
    name: "Anugrah Narayan Magadh Medical College",
    type: "Medical College Hospital",
    address: "Gaya",
    state: "Bihar",
    district: "Gaya",
    phone: "+91-0631-2222222",
    emergencyPhone: "+91-0631-2222000",
    website: "https://www.anmmcgaya.org/",
    hours: "24/7 Emergency Services",
    services: "TB care, DOTS therapy, emergency services",
    category: "govt"
  },

  // Chhattisgarh - Raipur
  {
    name: "Dr. Bhim Rao Ambedkar Memorial Hospital",
    type: "Government Hospital",
    address: "Raipur",
    state: "Chhattisgarh",
    district: "Raipur",
    phone: "+91-0771-2254000",
    emergencyPhone: "+91-0771-2254100",
    website: "https://www.cghealth.nic.in/",
    hours: "24/7 Emergency Services",
    services: "Free TB testing, DOTS centers, chest clinic",
    category: "govt"
  },
  {
    name: "MMI Narayana Multispeciality Hospital",
    type: "Private Hospital",
    address: "Lalpur, Raipur",
    state: "Chhattisgarh",
    district: "Raipur",
    phone: "+91-0771-4091000",
    emergencyPhone: "+91-0771-4091111",
    website: "https://www.narayanahealth.org/",
    hours: "24/7 Services",
    services: "TB treatment, pulmonology, ICU",
    category: "private"
  },

  // Delhi - South Delhi
  {
    name: "AIIMS Delhi",
    type: "Medical College Hospital",
    address: "Ansari Nagar, New Delhi",
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
    address: "Ring Road, New Delhi",
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
    name: "Max Super Speciality Hospital Saket",
    type: "Private Hospital",
    address: "Press Enclave Road, Saket",
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
    address: "Okhla Road, New Delhi",
    state: "Delhi",
    district: "South Delhi",
    phone: "+91-011-4713-5000",
    emergencyPhone: "+91-011-4713-5555",
    website: "https://www.fortishealthcare.com/",
    hours: "24/7 Services",
    services: "TB treatment, chest medicine, ICU",
    category: "private"
  },
  {
    name: "Apollo Hospitals Sarita Vihar",
    type: "Private Hospital",
    address: "Mathura Road, Sarita Vihar",
    state: "Delhi",
    district: "South Delhi",
    phone: "+91-011-2692-5858",
    emergencyPhone: "+91-011-2692-5000",
    website: "https://www.apollohospitals.com/",
    hours: "24/7 Services",
    services: "Advanced TB diagnostics, specialist consultation",
    category: "private"
  },

  // Delhi - Central Delhi
  {
    name: "Lady Hardinge Medical College",
    type: "Medical College Hospital",
    address: "Connaught Place",
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
    state: "Delhi",
    district: "Central Delhi",
    phone: "+91-011-2336-5525",
    emergencyPhone: "+91-011-2336-5000",
    website: "https://www.rmlh.nic.in/",
    hours: "24/7 Emergency Services",
    services: "Free TB testing, DOTS therapy",
    category: "govt"
  },

  // Gujarat - Ahmedabad
  {
    name: "Civil Hospital Ahmedabad",
    type: "Government Hospital",
    address: "Asarwa",
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
    name: "B.J. Medical College",
    type: "Medical College Hospital",
    address: "Asarwa, Ahmedabad",
    state: "Gujarat",
    district: "Ahmedabad",
    phone: "+91-079-2268-4400",
    emergencyPhone: "+91-079-2268-4000",
    website: "https://www.bjmcahd.org/",
    hours: "24/7 Emergency Services",
    services: "TB care, MDR-TB treatment, research",
    category: "govt"
  },
  {
    name: "Sterling Hospital",
    type: "Private Hospital",
    address: "Off Gurukul Road, Memnagar",
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
    name: "Apollo Hospitals Ahmedabad",
    type: "Private Hospital",
    address: "GIDC Estate, Bhat",
    state: "Gujarat",
    district: "Ahmedabad",
    phone: "+91-079-6670-1234",
    emergencyPhone: "+91-079-6670-1000",
    website: "https://www.apollohospitals.com/",
    hours: "24/7 Services",
    services: "Advanced TB diagnostics, specialist consultation",
    category: "private"
  },

  // Gujarat - Surat
  {
    name: "SMIMER Hospital",
    type: "Medical College Hospital",
    address: "Surat",
    state: "Gujarat",
    district: "Surat",
    phone: "+91-0261-2470751",
    emergencyPhone: "+91-0261-2470000",
    website: "https://www.smimer.org/",
    hours: "24/7 Emergency Services",
    services: "TB care, DOTS therapy, pulmonology",
    category: "govt"
  },
  {
    name: "Mahavir Hospital",
    type: "Private Hospital",
    address: "Ring Road, Surat",
    state: "Gujarat",
    district: "Surat",
    phone: "+91-0261-2470000",
    emergencyPhone: "+91-0261-2470100",
    website: "https://www.mahavirhospital.co.in/",
    hours: "24/7 Services",
    services: "TB diagnostics, chest medicine",
    category: "private"
  },

  // Haryana - Gurugram
  {
    name: "Civil Hospital Gurugram",
    type: "Government Hospital",
    address: "Sector 10, Gurugram",
    state: "Haryana",
    district: "Gurugram",
    phone: "+91-0124-2384444",
    emergencyPhone: "+91-0124-2384000",
    website: "https://www.nhmharyana.gov.in/",
    hours: "24/7 Emergency Services",
    services: "Free TB testing, DOTS centers",
    category: "govt"
  },
  {
    name: "Medanta - The Medicity",
    type: "Private Hospital",
    address: "Sector 38, Gurugram",
    state: "Haryana",
    district: "Gurugram",
    phone: "+91-0124-4141414",
    emergencyPhone: "+91-0124-4141000",
    website: "https://www.medanta.org/",
    hours: "24/7 Services",
    services: "Advanced TB care, pulmonology, ICU",
    category: "private"
  },
  {
    name: "Artemis Hospital",
    type: "Private Hospital",
    address: "Sector 51, Gurugram",
    state: "Haryana",
    district: "Gurugram",
    phone: "+91-0124-4511111",
    emergencyPhone: "+91-0124-4511000",
    website: "https://www.artemishospitals.com/",
    hours: "24/7 Services",
    services: "TB treatment, specialist consultation",
    category: "private"
  },

  // Haryana - Faridabad
  {
    name: "Sarvodaya Hospital",
    type: "Private Hospital",
    address: "Sector 8, Faridabad",
    state: "Haryana",
    district: "Faridabad",
    phone: "+91-0129-4197000",
    emergencyPhone: "+91-0129-4197100",
    website: "https://www.sarvodayahospital.com/",
    hours: "24/7 Services",
    services: "TB diagnostics, pulmonology",
    category: "private"
  },

  // Karnataka - Bengaluru Urban
  {
    name: "Victoria Hospital (BMCRI)",
    type: "Medical College Hospital",
    address: "Fort, K.R. Road, Bangalore",
    state: "Karnataka",
    district: "Bengaluru Urban",
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
    address: "Magadi Road, Rajajinagar",
    state: "Karnataka",
    district: "Bengaluru Urban",
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
    state: "Karnataka",
    district: "Bengaluru Urban",
    phone: "+91-080-2502-4444",
    emergencyPhone: "+91-080-2502-4000",
    website: "https://www.manipalhospitals.com/",
    hours: "24/7 Services",
    services: "TB diagnostics, pulmonology, specialist care",
    category: "private"
  },
  {
    name: "Apollo Hospitals Bannerghatta",
    type: "Private Hospital",
    address: "154/11, Bannerghatta Road",
    state: "Karnataka",
    district: "Bengaluru Urban",
    phone: "+91-080-2630-2244",
    emergencyPhone: "+91-080-2630-2000",
    website: "https://www.apollohospitals.com/",
    hours: "24/7 Services",
    services: "Advanced TB care, radiology, ICU",
    category: "private"
  },
  {
    name: "Narayana Health City",
    type: "Private Hospital",
    address: "258/A, Bommasandra, Bangalore",
    state: "Karnataka",
    district: "Bengaluru Urban",
    phone: "+91-080-7122-0000",
    emergencyPhone: "+91-080-7122-2000",
    website: "https://www.narayanahealth.org/",
    hours: "24/7 Services",
    services: "TB treatment, pulmonology department",
    category: "private"
  },
  {
    name: "Fortis Hospital",
    type: "Private Hospital",
    address: "14, Cunningham Road",
    state: "Karnataka",
    district: "Bengaluru Urban",
    phone: "+91-080-6621-4444",
    emergencyPhone: "+91-080-6621-4000",
    website: "https://www.fortishealthcare.com/",
    hours: "24/7 Services",
    services: "Comprehensive TB care, specialist consultation",
    category: "private"
  },

  // Karnataka - Mysuru
  {
    name: "K.R. Hospital Mysore",
    type: "Government Hospital",
    address: "Sayyaji Rao Road, Mysore",
    state: "Karnataka",
    district: "Mysuru",
    phone: "+91-0821-2420302",
    emergencyPhone: "+91-0821-2420000",
    website: "https://www.krhospital.kar.nic.in/",
    hours: "24/7 Emergency Services",
    services: "TB care, DOTS therapy, emergency services",
    category: "govt"
  },

  // Kerala - Thiruvananthapuram
  {
    name: "Government Medical College Thiruvananthapuram",
    type: "Medical College Hospital",
    address: "Medical College, Thiruvananthapuram",
    state: "Kerala",
    district: "Thiruvananthapuram",
    phone: "+91-0471-2443152",
    emergencyPhone: "+91-0471-2443000",
    website: "https://www.tvmmc.gov.in/",
    hours: "24/7 Emergency Services",
    services: "Comprehensive TB care, MDR-TB treatment",
    category: "govt"
  },
  {
    name: "KIMS Hospital",
    type: "Private Hospital",
    address: "PB No.1, Trivandrum",
    state: "Kerala",
    district: "Thiruvananthapuram",
    phone: "+91-0471-3041000",
    emergencyPhone: "+91-0471-3041100",
    website: "https://www.kimsglobal.com/",
    hours: "24/7 Services",
    services: "TB diagnostics, pulmonology, specialist care",
    category: "private"
  },

  // Kerala - Ernakulam (Kochi)
  {
    name: "Ernakulam Medical Centre",
    type: "Government Hospital",
    address: "Palarivattom, Kochi",
    state: "Kerala",
    district: "Ernakulam",
    phone: "+91-0484-2803000",
    emergencyPhone: "+91-0484-2803100",
    website: "https://www.health.kerala.gov.in/",
    hours: "24/7 Emergency Services",
    services: "TB care, DOTS therapy, emergency care",
    category: "govt"
  },
  {
    name: "Amrita Institute of Medical Sciences",
    type: "Medical College Hospital",
    address: "AIMS Ponekkara P.O, Kochi",
    state: "Kerala",
    district: "Ernakulam",
    phone: "+91-0484-2851234",
    emergencyPhone: "+91-0484-2851000",
    website: "https://www.aimshospitals.org/",
    hours: "24/7 Services",
    services: "Advanced TB care, research, pulmonology",
    category: "private"
  },
  {
    name: "Lakeshore Hospital",
    type: "Private Hospital",
    address: "NH-47 Bypass, Maradu, Kochi",
    state: "Kerala",
    district: "Ernakulam",
    phone: "+91-0484-2701441",
    emergencyPhone: "+91-0484-2701000",
    website: "https://www.lakeshorehospital.com/",
    hours: "24/7 Services",
    services: "TB treatment, chest medicine, ICU",
    category: "private"
  },

  // Kerala - Kozhikode (Calicut)
  {
    name: "Government Medical College Kozhikode",
    type: "Medical College Hospital",
    address: "Kozhikode",
    state: "Kerala",
    district: "Kozhikode",
    phone: "+91-0495-2359500",
    emergencyPhone: "+91-0495-2359000",
    website: "https://www.mckozhikode.kerala.gov.in/",
    hours: "24/7 Emergency Services",
    services: "TB care, DOTS therapy, MDR-TB treatment",
    category: "govt"
  },

  // Madhya Pradesh - Bhopal
  {
    name: "Hamidia Hospital",
    type: "Government Hospital",
    address: "Sultania Road, Bhopal",
    state: "Madhya Pradesh",
    district: "Bhopal",
    phone: "+91-0755-2740012",
    emergencyPhone: "+91-0755-2740000",
    website: "https://www.mphealth.nic.in/",
    hours: "24/7 Emergency Services",
    services: "Free TB testing, DOTS centers, chest clinic",
    category: "govt"
  },
  {
    name: "Gandhi Medical College Hospital",
    type: "Medical College Hospital",
    address: "Sultania Road, Bhopal",
    state: "Madhya Pradesh",
    district: "Bhopal",
    phone: "+91-0755-2707101",
    emergencyPhone: "+91-0755-2707000",
    website: "https://www.gmcbhopal.edu.in/",
    hours: "24/7 Emergency Services",
    services: "TB care, MDR-TB treatment, pulmonology",
    category: "govt"
  },
  {
    name: "Bansal Hospital",
    type: "Private Hospital",
    address: "Raisen Bypass Road, Bhopal",
    state: "Madhya Pradesh",
    district: "Bhopal",
    phone: "+91-0755-4908000",
    emergencyPhone: "+91-0755-4908100",
    website: "https://www.bansalhospital.com/",
    hours: "24/7 Services",
    services: "TB diagnostics, pulmonology, specialist care",
    category: "private"
  },

  // Madhya Pradesh - Indore
  {
    name: "Maharaja Yeshwantrao Hospital (MY Hospital)",
    type: "Government Hospital",
    address: "MG Road, Indore",
    state: "Madhya Pradesh",
    district: "Indore",
    phone: "+91-0731-2471877",
    emergencyPhone: "+91-0731-2471000",
    website: "https://www.myhindore.org/",
    hours: "24/7 Emergency Services",
    services: "TB care, DOTS therapy, emergency services",
    category: "govt"
  },
  {
    name: "Bombay Hospital Indore",
    type: "Private Hospital",
    address: "Indore",
    state: "Madhya Pradesh",
    district: "Indore",
    phone: "+91-0731-4222222",
    emergencyPhone: "+91-0731-4222000",
    website: "https://www.bombayhospitalindore.com/",
    hours: "24/7 Services",
    services: "TB treatment, pulmonology, ICU",
    category: "private"
  },

  // Madhya Pradesh - Jabalpur
  {
    name: "Netaji Subhash Chandra Bose Medical College",
    type: "Medical College Hospital",
    address: "Jabalpur",
    state: "Madhya Pradesh",
    district: "Jabalpur",
    phone: "+91-0761-2671281",
    emergencyPhone: "+91-0761-2671000",
    website: "https://www.nscbmcjabalpur.org/",
    hours: "24/7 Emergency Services",
    services: "TB care, DOTS therapy, chest medicine",
    category: "govt"
  },

  // Maharashtra - Mumbai City
  {
    name: "KEM Hospital Mumbai",
    type: "Medical College Hospital",
    address: "489, Sardar Moodliar Road, Parel",
    state: "Maharashtra",
    district: "Mumbai City",
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
    state: "Maharashtra",
    district: "Mumbai City",
    phone: "+91-022-2407-6101",
    emergencyPhone: "+91-022-2407-6000",
    website: "https://www.sionhospital.org/",
    hours: "24/7 Emergency Services",
    services: "TB care, DOTS therapy, chest clinic",
    category: "govt"
  },

  // Maharashtra - Mumbai Suburban
  {
    name: "Lilavati Hospital",
    type: "Private Hospital",
    address: "A-791, Bandra Reclamation",
    state: "Maharashtra",
    district: "Mumbai Suburban",
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
    state: "Maharashtra",
    district: "Mumbai Suburban",
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
    state: "Maharashtra",
    district: "Mumbai Suburban",
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
    address: "40, Sassoon Road, Pune",
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
    address: "Erandwane, Pune",
    state: "Maharashtra",
    district: "Pune",
    phone: "+91-020-6710-3000",
    emergencyPhone: "+91-020-6710-3333",
    website: "https://www.dmhospital.org/",
    hours: "24/7 Services",
    services: "TB treatment, chest medicine",
    category: "private"
  },
  {
    name: "Jehangir Hospital",
    type: "Private Hospital",
    address: "32, Sassoon Road, Pune",
    state: "Maharashtra",
    district: "Pune",
    phone: "+91-020-2605-2999",
    emergencyPhone: "+91-020-2605-2000",
    website: "https://www.jehangirhospital.com/",
    hours: "24/7 Services",
    services: "TB care, specialist consultation, ICU",
    category: "private"
  },

  // Maharashtra - Nagpur
  {
    name: "Government Medical College Nagpur",
    type: "Medical College Hospital",
    address: "Medical Square, Nagpur",
    state: "Maharashtra",
    district: "Nagpur",
    phone: "+91-0712-2740000",
    emergencyPhone: "+91-0712-2740100",
    website: "https://www.gmcnagpur.ac.in/",
    hours: "24/7 Emergency Services",
    services: "TB care, MDR-TB treatment, DOTS therapy",
    category: "govt"
  },
  {
    name: "Orange City Hospital",
    type: "Private Hospital",
    address: "Wardha Road, Nagpur",
    state: "Maharashtra",
    district: "Nagpur",
    phone: "+91-0712-6640000",
    emergencyPhone: "+91-0712-6640100",
    website: "https://www.orangecityhospital.com/",
    hours: "24/7 Services",
    services: "TB diagnostics, pulmonology, specialist care",
    category: "private"
  },

  // Odisha - Khordha (Bhubaneswar)
  {
    name: "Capital Hospital Bhubaneswar",
    type: "Government Hospital",
    address: "Unit 6, Bhubaneswar",
    state: "Odisha",
    district: "Khordha",
    phone: "+91-0674-2393111",
    emergencyPhone: "+91-0674-2393000",
    website: "https://www.odishahealth.gov.in/",
    hours: "24/7 Emergency Services",
    services: "Free TB testing, DOTS centers, emergency care",
    category: "govt"
  },
  {
    name: "AIIMS Bhubaneswar",
    type: "Medical College Hospital",
    address: "Sijua, Bhubaneswar",
    state: "Odisha",
    district: "Khordha",
    phone: "+91-0674-2476200",
    emergencyPhone: "+91-0674-2476000",
    website: "https://www.aiimsbhubaneswar.nic.in/",
    hours: "24/7 Emergency Services",
    services: "Advanced TB care, research, MDR-TB treatment",
    category: "govt"
  },
  {
    name: "Apollo Hospitals Bhubaneswar",
    type: "Private Hospital",
    address: "Sainik School Post, Bhubaneswar",
    state: "Odisha",
    district: "Khordha",
    phone: "+91-0674-666-6668",
    emergencyPhone: "+91-0674-666-6000",
    website: "https://www.apollohospitals.com/",
    hours: "24/7 Services",
    services: "TB treatment, pulmonology, ICU",
    category: "private"
  },

  // Odisha - Cuttack
  {
    name: "SCB Medical College Hospital",
    type: "Medical College Hospital",
    address: "Mangalabag, Cuttack",
    state: "Odisha",
    district: "Cuttack",
    phone: "+91-0671-2302141",
    emergencyPhone: "+91-0671-2302000",
    website: "https://www.scbmch.ac.in/",
    hours: "24/7 Emergency Services",
    services: "Comprehensive TB care, DOTS therapy, research",
    category: "govt"
  },

  // Punjab - Ludhiana
  {
    name: "Christian Medical College Ludhiana",
    type: "Medical College Hospital",
    address: "Brown Road, Ludhiana",
    state: "Punjab",
    district: "Ludhiana",
    phone: "+91-0161-2606700",
    emergencyPhone: "+91-0161-2606000",
    website: "https://www.cmcludhiana.in/",
    hours: "24/7 Emergency Services",
    services: "TB care, MDR-TB treatment, pulmonology",
    category: "private"
  },
  {
    name: "Dayanand Medical College Hospital",
    type: "Medical College Hospital",
    address: "Tagore Nagar, Ludhiana",
    state: "Punjab",
    district: "Ludhiana",
    phone: "+91-0161-2302031",
    emergencyPhone: "+91-0161-2302000",
    website: "https://www.dmch.edu/",
    hours: "24/7 Emergency Services",
    services: "TB care, DOTS therapy, chest medicine",
    category: "private"
  },

  // Punjab - Amritsar
  {
    name: "Guru Nanak Dev Hospital",
    type: "Government Hospital",
    address: "Medical College Road, Amritsar",
    state: "Punjab",
    district: "Amritsar",
    phone: "+91-0183-2502400",
    emergencyPhone: "+91-0183-2502000",
    website: "https://www.gndhospital.amritsar.nic.in/",
    hours: "24/7 Emergency Services",
    services: "Free TB testing, DOTS therapy, emergency care",
    category: "govt"
  },

  // Rajasthan - Jaipur South
  {
    name: "SMS Hospital Jaipur",
    type: "Government Hospital",
    address: "JLN Marg, Jaipur",
    state: "Rajasthan",
    district: "Jaipur South",
    phone: "+91-0141-2560291",
    emergencyPhone: "+91-0141-2560000",
    website: "https://www.smshospitaljaipur.nic.in/",
    hours: "24/7 Emergency Services",
    services: "Comprehensive TB care, DOTS therapy, emergency",
    category: "govt"
  },
  {
    name: "Fortis Escorts Hospital Jaipur",
    type: "Private Hospital",
    address: "Jawahar Lal Nehru Marg, Jaipur",
    state: "Rajasthan",
    district: "Jaipur South",
    phone: "+91-0141-254-7000",
    emergencyPhone: "+91-0141-254-7100",
    website: "https://www.fortishealthcare.com/",
    hours: "24/7 Services",
    services: "TB diagnostics, pulmonology, specialist care",
    category: "private"
  },
  {
    name: "Narayana Multispeciality Hospital",
    type: "Private Hospital",
    address: "Sector 28, Pratap Nagar, Jaipur",
    state: "Rajasthan",
    district: "Jaipur South",
    phone: "+91-0141-7150000",
    emergencyPhone: "+91-0141-7150100",
    website: "https://www.narayanahealth.org/",
    hours: "24/7 Services",
    services: "TB treatment, chest medicine, ICU",
    category: "private"
  },

  // Tamil Nadu - Chennai
  {
    name: "Government Stanley Medical College",
    type: "Medical College Hospital",
    address: "Old Jail Road, Chennai",
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
    address: "Tambaram Sanatorium, Chennai",
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
    name: "Apollo Hospitals Greams Road",
    type: "Private Hospital",
    address: "21, Greams Lane, Chennai",
    state: "Tamil Nadu",
    district: "Chennai",
    phone: "+91-044-2829-0200",
    emergencyPhone: "+91-044-2829-0000",
    website: "https://www.apollohospitals.com/",
    hours: "24/7 Services",
    services: "Advanced TB diagnostics, pulmonology",
    category: "private"
  },
  {
    name: "Fortis Malar Hospital",
    type: "Private Hospital",
    address: "52, 1st Main Road, Gandhi Nagar, Adyar",
    state: "Tamil Nadu",
    district: "Chennai",
    phone: "+91-044-4289-2222",
    emergencyPhone: "+91-044-4289-2000",
    website: "https://www.fortishealthcare.com/",
    hours: "24/7 Services",
    services: "TB treatment, chest medicine",
    category: "private"
  },

  // Tamil Nadu - Coimbatore
  {
    name: "Coimbatore Medical College Hospital",
    type: "Medical College Hospital",
    address: "Avinashi Road, Coimbatore",
    state: "Tamil Nadu",
    district: "Coimbatore",
    phone: "+91-0422-2204200",
    emergencyPhone: "+91-0422-2204000",
    website: "https://www.cmch.ac.in/",
    hours: "24/7 Emergency Services",
    services: "TB care, DOTS therapy, MDR-TB treatment",
    category: "govt"
  },
  {
    name: "PSG Hospitals",
    type: "Private Hospital",
    address: "Avinashi Road, Coimbatore",
    state: "Tamil Nadu",
    district: "Coimbatore",
    phone: "+91-0422-4344344",
    emergencyPhone: "+91-0422-4344000",
    website: "https://www.psgimsr.ac.in/",
    hours: "24/7 Services",
    services: "TB diagnostics, pulmonology, specialist care",
    category: "private"
  },

  // Tamil Nadu - Madurai
  {
    name: "Government Rajaji Hospital",
    type: "Government Hospital",
    address: "Park Town, Madurai",
    state: "Tamil Nadu",
    district: "Madurai",
    phone: "+91-0452-2530644",
    emergencyPhone: "+91-0452-2530000",
    website: "https://www.grhmadurai.org/",
    hours: "24/7 Emergency Services",
    services: "Free TB care, DOTS therapy, emergency",
    category: "govt"
  },

  // Telangana - Hyderabad
  {
    name: "Osmania General Hospital",
    type: "Government Hospital",
    address: "Afzal Gunj, Hyderabad",
    state: "Telangana",
    district: "Hyderabad",
    phone: "+91-040-2461-3333",
    emergencyPhone: "+91-040-2461-3000",
    website: "https://www.ogh.gov.in/",
    hours: "24/7 Emergency Services",
    services: "Comprehensive TB care, DOTS therapy, free treatment",
    category: "govt"
  },
  {
    name: "Nizam's Institute of Medical Sciences (NIMS)",
    type: "Medical College Hospital",
    address: "Punjagutta, Hyderabad",
    state: "Telangana",
    district: "Hyderabad",
    phone: "+91-040-2348-9000",
    emergencyPhone: "+91-040-2348-9100",
    website: "https://www.nims.edu.in/",
    hours: "24/7 Emergency Services",
    services: "TB care, MDR-TB treatment, research",
    category: "govt"
  },
  {
    name: "Apollo Hospitals Jubilee Hills",
    type: "Private Hospital",
    address: "Film Nagar, Jubilee Hills, Hyderabad",
    state: "Telangana",
    district: "Hyderabad",
    phone: "+91-040-2360-7777",
    emergencyPhone: "+91-040-2360-7000",
    website: "https://www.apollohospitals.com/",
    hours: "24/7 Services",
    services: "Advanced TB care, pulmonology, ICU",
    category: "private"
  },
  {
    name: "Yashoda Hospital",
    type: "Private Hospital",
    address: "Malakpet, Hyderabad",
    state: "Telangana",
    district: "Hyderabad",
    phone: "+91-040-4445-5000",
    emergencyPhone: "+91-040-4445-5100",
    website: "https://www.yashodahospitals.com/",
    hours: "24/7 Services",
    services: "TB treatment, specialist consultation",
    category: "private"
  },

  // Telangana - Rangareddy
  {
    name: "Continental Hospitals",
    type: "Private Hospital",
    address: "IT Park Road, Gachibowli, Hyderabad",
    state: "Telangana",
    district: "Rangareddy",
    phone: "+91-040-6767-1000",
    emergencyPhone: "+91-040-6767-1111",
    website: "https://www.continentalhospitals.com/",
    hours: "24/7 Services",
    services: "TB diagnostics, pulmonology, specialist care",
    category: "private"
  },

  // Uttar Pradesh - Lucknow
  {
    name: "King George's Medical University",
    type: "Medical College Hospital",
    address: "Shah Mina Road, Lucknow",
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
    name: "Sahara Hospital Lucknow",
    type: "Private Hospital",
    address: "Viraj Khand, Gomti Nagar, Lucknow",
    state: "Uttar Pradesh",
    district: "Lucknow",
    phone: "+91-0522-398-8888",
    emergencyPhone: "+91-0522-398-8000",
    website: "https://www.saharahospital.com/",
    hours: "24/7 Services",
    services: "TB diagnostics, pulmonology department",
    category: "private"
  },
  {
    name: "Medanta Lucknow",
    type: "Private Hospital",
    address: "Sector B, Pocket 1, Gomti Nagar Extension",
    state: "Uttar Pradesh",
    district: "Lucknow",
    phone: "+91-0522-4141414",
    emergencyPhone: "+91-0522-4141000",
    website: "https://www.medanta.org/",
    hours: "24/7 Services",
    services: "TB treatment, chest medicine, ICU",
    category: "private"
  },

  // Uttar Pradesh - Varanasi
  {
    name: "Sir Sunderlal Hospital (BHU)",
    type: "Medical College Hospital",
    address: "Banaras Hindu University, Varanasi",
    state: "Uttar Pradesh",
    district: "Varanasi",
    phone: "+91-0542-2369444",
    emergencyPhone: "+91-0542-2369000",
    website: "https://www.bhu.ac.in/",
    hours: "24/7 Emergency Services",
    services: "TB care, DOTS therapy, chest medicine",
    category: "govt"
  },

  // Uttar Pradesh - Agra
  {
    name: "Sarojini Naidu Medical College",
    type: "Medical College Hospital",
    address: "Hospital Road, Agra",
    state: "Uttar Pradesh",
    district: "Agra",
    phone: "+91-0562-2261803",
    emergencyPhone: "+91-0562-2261000",
    website: "https://www.snmc.ac.in/",
    hours: "24/7 Emergency Services",
    services: "TB care, DOTS therapy, pulmonology",
    category: "govt"
  },

  // Uttar Pradesh - Kanpur Nagar
  {
    name: "LPS Institute of Cardiology (Ganesh Shankar Vidyarthi Memorial)",
    type: "Government Hospital",
    address: "Kanpur",
    state: "Uttar Pradesh",
    district: "Kanpur Nagar",
    phone: "+91-0512-2556100",
    emergencyPhone: "+91-0512-2556000",
    website: "https://www.lpsic.org/",
    hours: "24/7 Emergency Services",
    services: "TB screening, emergency care",
    category: "govt"
  },

  // Uttarakhand - Dehradun
  {
    name: "Doon Hospital",
    type: "Government Hospital",
    address: "Prem Nagar, Dehradun",
    state: "Uttarakhand",
    district: "Dehradun",
    phone: "+91-0135-2677777",
    emergencyPhone: "+91-0135-2677000",
    website: "https://www.doonhospital.gov.in/",
    hours: "24/7 Emergency Services",
    services: "TB care, DOTS therapy, emergency services",
    category: "govt"
  },
  {
    name: "Max Super Speciality Hospital Dehradun",
    type: "Private Hospital",
    address: "Chakrata Road, Dehradun",
    state: "Uttarakhand",
    district: "Dehradun",
    phone: "+91-0135-6687000",
    emergencyPhone: "+91-0135-6687100",
    website: "https://www.maxhealthcare.in/",
    hours: "24/7 Services",
    services: "TB diagnostics, pulmonology, specialist care",
    category: "private"
  },

  // West Bengal - Kolkata
  {
    name: "SSKM Hospital (Kolkata Medical College)",
    type: "Medical College Hospital",
    address: "244 AJC Bose Road, Kolkata",
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
    address: "1, Khudiram Bose Sarani, Kolkata",
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
    name: "Apollo Gleneagles Hospitals",
    type: "Private Hospital",
    address: "58, Canal Circular Road, Kadapara",
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
    name: "Fortis Hospital Anandapur",
    type: "Private Hospital",
    address: "730, Anandapur, EM Bypass Road",
    state: "West Bengal",
    district: "Kolkata",
    phone: "+91-033-6628-4444",
    emergencyPhone: "+91-033-6628-4000",
    website: "https://www.fortishealthcare.com/",
    hours: "24/7 Services",
    services: "TB treatment, pulmonology, radiology, ICU",
    category: "private"
  },
  {
    name: "Peerless Hospital",
    type: "Private Hospital",
    address: "360, Panchasayar, Garia",
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
    name: "Rabindranath Tagore International Institute of Cardiac Sciences",
    type: "Private Hospital",
    address: "124, Mukundapur, E.M Bypass, Kolkata",
    state: "West Bengal",
    district: "Kolkata",
    phone: "+91-033-6606-3800",
    emergencyPhone: "+91-033-6606-3000",
    website: "https://www.rtiics.org/",
    hours: "24/7 Services",
    services: "TB screening, chest diagnostics",
    category: "private"
  },

  // West Bengal - North 24 Parganas
  {
    name: "Barasat District Hospital",
    type: "Government Hospital",
    address: "Barasat Station Road",
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
    name: "ILS Hospital Salt Lake",
    type: "Private Hospital",
    address: "DD-6, Sector 1, Salt Lake",
    state: "West Bengal",
    district: "North 24 Parganas",
    phone: "+91-033-4040-8080",
    emergencyPhone: "+91-033-4040-8000",
    website: "https://www.ilshospitals.com/",
    hours: "24/7 Services",
    services: "TB diagnostics, pulmonology, ICU",
    category: "private"
  },

  // West Bengal - Murshidabad
  {
    name: "Murshidabad Medical College & Hospital",
    type: "Medical College Hospital",
    address: "Berhampore College Road, Berhampore",
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
    type: "District Hospital",
    address: "Civil Hospital Compound, Berhampore",
    state: "West Bengal",
    district: "Murshidabad",
    phone: "+91-03482-250123",
    emergencyPhone: "+91-03482-250200",
    website: "https://www.wbhealth.gov.in/",
    hours: "Mon-Sat: 9:00 AM - 5:00 PM",
    services: "Free TB testing, DOTS centers, X-ray",
    category: "govt"
  },

  // West Bengal - Howrah
  {
    name: "Howrah General Hospital",
    type: "Government Hospital",
    address: "College Street, Howrah",
    state: "West Bengal",
    district: "Howrah",
    phone: "+91-033-2660-4141",
    emergencyPhone: "+91-033-2660-4000",
    website: "https://www.wbhealth.gov.in/",
    hours: "24/7 Emergency Services",
    services: "Free TB testing, DOTS centers",
    category: "govt"
  },
];

// Helper functions
export const getHospitalsByStateDistrict = (state: string, district: string): Hospital[] => {
  return expandedHospitalDatabase.filter(h => h.state === state && h.district === district);
};

export const getHospitalsByState = (state: string): Hospital[] => {
  return expandedHospitalDatabase.filter(h => h.state === state);
};

export const getAllHospitals = (): Hospital[] => {
  return expandedHospitalDatabase;
};
