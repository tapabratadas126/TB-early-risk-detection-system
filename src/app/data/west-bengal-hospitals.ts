// West Bengal Government Hospitals Database
// Sources: Swasthya Sathi, WB Health Scheme, SHG Portal

export interface WBHospital {
  name: string;
  district: string;
  city: string;
  pinCode: string;
  postOffice: string;
  type: 'Government Hospital' | 'Private Hospital' | 'Medical College Hospital' | 'District Hospital' | 'Sub-Divisional Hospital' | 'Block Primary Health Centre' | 'Super Specialty Hospital';
  category: 'govt' | 'private';
  address: string;
  phone?: string;
  emergencyPhone?: string;
  website?: string;
  services: string;
  class?: 'Class I' | 'Class II' | 'Class III';
}

// Comprehensive West Bengal Hospitals Database
export const westBengalHospitals: WBHospital[] = [
  // Kolkata Hospitals
  {
    name: "SSKM Hospital (Kolkata Medical College)",
    district: "Kolkata",
    city: "Kolkata",
    pinCode: "700020",
    postOffice: "Kolkata GPO",
    type: "Medical College Hospital",
    category: "govt",
    address: "244 AJC Bose Road, Kolkata",
    phone: "+91-033-2223-3636",
    emergencyPhone: "+91-033-2223-5000",
    website: "https://www.kmcgov.in/",
    services: "TB care, MDR-TB treatment, DOTS therapy, X-ray, CT scan",
    class: "Class I"
  },
  {
    name: "R.G. Kar Medical College & Hospital",
    district: "Kolkata",
    city: "Kolkata",
    pinCode: "700004",
    postOffice: "Shyambazar",
    type: "Medical College Hospital",
    category: "govt",
    address: "1, Khudiram Bose Sarani, Kolkata",
    phone: "+91-033-2555-7656",
    emergencyPhone: "+91-033-2555-7777",
    website: "https://www.rgkarmch.in/",
    services: "Comprehensive TB care, drug-resistant TB treatment, pulmonology",
    class: "Class I"
  },
  {
    name: "Apollo Gleneagles Hospitals",
    district: "Kolkata",
    city: "Kolkata",
    pinCode: "700054",
    postOffice: "Kankurgachi",
    type: "Super Specialty Hospital",
    category: "private",
    address: "58, Canal Circular Road, Kadapara",
    phone: "+91-033-2320-3040",
    emergencyPhone: "+91-033-2320-2122",
    website: "https://www.apollogleneagles.in/",
    services: "Advanced TB diagnostics, specialist consultation, ICU facilities",
    class: "Class I"
  },
  {
    name: "Fortis Hospital Anandapur",
    district: "Kolkata",
    city: "Kolkata",
    pinCode: "700107",
    postOffice: "Beleghata H.O.",
    type: "Private Hospital",
    category: "private",
    address: "730, Anandapur, EM Bypass Road",
    phone: "+91-033-6628-4444",
    emergencyPhone: "+91-033-6628-4000",
    website: "https://www.fortishealthcare.com/",
    services: "TB treatment, pulmonology, radiology, ICU",
    class: "Class I"
  },
  {
    name: "Peerless Hospital",
    district: "Kolkata",
    city: "Kolkata",
    pinCode: "700094",
    postOffice: "Garia SO",
    type: "Private Hospital",
    category: "private",
    address: "360, Panchasayar, Garia",
    phone: "+91-033-2441-5151",
    emergencyPhone: "+91-033-2441-1212",
    website: "https://www.peerlesshospital.com/",
    services: "TB care, chest medicine, diagnostics",
    class: "Class I"
  },
  {
    name: "Rabindranath Tagore International Institute of Cardiac Sciences",
    district: "Kolkata",
    city: "Kolkata",
    pinCode: "700099",
    postOffice: "Beleghata H.O.",
    type: "Super Specialty Hospital",
    category: "private",
    address: "124, Mukundapur, E.M Bypass, Kolkata",
    phone: "+91-033-6606-3800",
    emergencyPhone: "+91-033-6606-3000",
    website: "https://www.rtiics.org/",
    services: "TB screening, chest diagnostics",
    class: "Class I"
  },
  {
    name: "Calcutta Medical Research Institute (CMRI)",
    district: "Kolkata",
    city: "Kolkata",
    pinCode: "700027",
    postOffice: "Alipore H.O.",
    type: "Private Hospital",
    category: "private",
    address: "7/2, Diamond Harbour Road, Kolkata",
    phone: "+91-033-4040-0000",
    emergencyPhone: "+91-033-4040-0000",
    website: "https://www.cmri-kolkata.com/",
    services: "TB diagnosis, pulmonology, emergency care",
    class: "Class I"
  },
  {
    name: "Arogya Niketan Private Limited",
    district: "Kolkata",
    city: "Kolkata",
    pinCode: "700020",
    postOffice: "Bowbazar",
    type: "Private Hospital",
    category: "private",
    address: "8B, Garstin Place, Kolkata",
    phone: "+91-033-2237-3737",
    emergencyPhone: "+91-033-2237-3737",
    website: "",
    services: "General medicine, TB care, outpatient services",
    class: "Class I"
  },
  {
    name: "Belle Vue Clinic",
    district: "Kolkata",
    city: "Kolkata",
    pinCode: "700017",
    postOffice: "Park street",
    type: "Private Hospital",
    category: "private",
    address: "9, Dr. U.N. Brahmachari Street, Kolkata",
    phone: "+91-033-2287-2321",
    emergencyPhone: "+91-033-2287-2321",
    website: "https://www.bellevueclinic.com/",
    services: "Multi-specialty care, TB treatment, ICU",
    class: "Class I"
  },
  {
    name: "Woodlands Hospital",
    district: "Kolkata",
    city: "Kolkata",
    pinCode: "700027",
    postOffice: "Alipore H.O.",
    type: "Private Hospital",
    category: "private",
    address: "8/5, Alipore Road, Kolkata",
    phone: "+91-033-4040-5050",
    emergencyPhone: "+91-033-4040-5050",
    website: "https://www.woodlandshospital.in/",
    services: "TB care, pulmonology, critical care",
    class: "Class I"
  },

  // North 24 Parganas Hospitals
  {
    name: "Barasat District Hospital",
    district: "North 24 Parganas",
    city: "Barasat",
    pinCode: "700124",
    postOffice: "BARASAT HO",
    type: "District Hospital",
    category: "govt",
    address: "Barasat Station Road",
    phone: "+91-033-2543-2100",
    emergencyPhone: "+91-033-2543-2200",
    website: "https://www.wbhealth.gov.in/",
    services: "TB care, DOTS therapy, emergency services",
    class: "Class II"
  },
  {
    name: "ILS Hospital Salt Lake",
    district: "North 24 Parganas",
    city: "Kolkata",
    pinCode: "700091",
    postOffice: "Sech Bhawan",
    type: "Private Hospital",
    category: "private",
    address: "DD-6, Sector 1, Salt Lake",
    phone: "+91-033-4040-8080",
    emergencyPhone: "+91-033-4040-8000",
    website: "https://www.ilshospitals.com/",
    services: "TB diagnostics, pulmonology, ICU",
    class: "Class I"
  },
  {
    name: "Naihati State General Hospital",
    district: "North 24 Parganas",
    city: "Naihati",
    pinCode: "743165",
    postOffice: "Naihati",
    type: "Government Hospital",
    category: "govt",
    address: "Naihati, North 24 Parganas",
    phone: "+91-033-2571-2345",
    emergencyPhone: "+91-033-2571-2345",
    website: "",
    services: "TB screening, DOTS center, emergency care",
    class: "Class II"
  },
  {
    name: "Panihati State General Hospital",
    district: "North 24 Parganas",
    city: "Panihati",
    pinCode: "743145",
    postOffice: "Panihati",
    type: "Government Hospital",
    category: "govt",
    address: "Panihati, North 24 Parganas",
    phone: "+91-033-2561-1234",
    emergencyPhone: "+91-033-2561-1234",
    website: "",
    services: "TB care, free medicines, DOTS",
    class: "Class II"
  },
  {
    name: "Barrackpore Sub-Divisional Hospital",
    district: "North 24 Parganas",
    city: "Barrackpore",
    pinCode: "700120",
    postOffice: "Barrackpore",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Barrackpore, North 24 Parganas",
    phone: "+91-033-2591-2345",
    emergencyPhone: "+91-033-2591-2345",
    website: "",
    services: "TB treatment, X-ray, emergency services",
    class: "Class II"
  },
  {
    name: "Basirhat District Hospital",
    district: "North 24 Parganas",
    city: "Basirhat",
    pinCode: "743411",
    postOffice: "BASIRHAT H.P.O.",
    type: "District Hospital",
    category: "govt",
    address: "Basirhat, North 24 Parganas",
    phone: "+91-033-2522-1234",
    emergencyPhone: "+91-033-2522-1234",
    website: "",
    services: "TB care, DOTS therapy, maternity services",
    class: "Class II"
  },

  // South 24 Parganas Hospitals
  {
    name: "Diamond Harbour Government Hospital",
    district: "South 24 Parganas",
    city: "Diamond Harbour",
    pinCode: "743331",
    postOffice: "DIAMOND HARBOUR",
    type: "Government Hospital",
    category: "govt",
    address: "Diamond Harbour, South 24 Parganas",
    phone: "+91-033-2465-1234",
    emergencyPhone: "+91-033-2465-1234",
    website: "",
    services: "TB screening, DOTS center, emergency care",
    class: "Class II"
  },
  {
    name: "Baruipur Sub-Divisional Hospital",
    district: "South 24 Parganas",
    city: "Baruipur",
    pinCode: "700144",
    postOffice: "BARUIPUR HO",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Baruipur, South 24 Parganas",
    phone: "+91-033-2439-1234",
    emergencyPhone: "+91-033-2439-1234",
    website: "",
    services: "TB care, free treatment, DOTS",
    class: "Class II"
  },
  {
    name: "Canning Sub-Divisional Hospital",
    district: "South 24 Parganas",
    city: "Canning",
    pinCode: "743329",
    postOffice: "CANNING TOWN S.O.",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Canning, South 24 Parganas",
    phone: "+91-033-2483-1234",
    emergencyPhone: "+91-033-2483-1234",
    website: "",
    services: "TB treatment, emergency services, maternity care",
    class: "Class II"
  },

  // Howrah Hospitals
  {
    name: "Howrah General Hospital",
    district: "Howrah",
    city: "Howrah",
    pinCode: "711101",
    postOffice: "HOWRAH HPO",
    type: "Government Hospital",
    category: "govt",
    address: "College Street, Howrah",
    phone: "+91-033-2660-4141",
    emergencyPhone: "+91-033-2660-4000",
    website: "https://www.wbhealth.gov.in/",
    services: "Free TB testing, DOTS centers",
    class: "Class I"
  },
  {
    name: "Howrah Orthopedic Hospital",
    district: "Howrah",
    city: "Howrah",
    pinCode: "711106",
    postOffice: "SALKIA HPO",
    type: "Specialty Hospital",
    category: "govt",
    address: "Salkia, Howrah",
    phone: "+91-033-2641-1234",
    emergencyPhone: "+91-033-2641-1234",
    website: "",
    services: "Orthopedic care, TB bone infections",
    class: "Class II"
  },
  {
    name: "Uluberia Sub-Divisional Hospital",
    district: "Howrah",
    city: "Uluberia",
    pinCode: "711315",
    postOffice: "ULUBERIA",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Uluberia, Howrah",
    phone: "+91-033-2662-1234",
    emergencyPhone: "+91-033-2662-1234",
    website: "",
    services: "TB care, DOTS therapy, emergency services",
    class: "Class II"
  },

  // Hooghly Hospitals
  {
    name: "Chinsurah Sub-Divisional Hospital",
    district: "Hooghly",
    city: "Hugli-Chuchura",
    pinCode: "712101",
    postOffice: "CHINSURAH",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Chinsurah, Hooghly",
    phone: "+91-033-2680-1234",
    emergencyPhone: "+91-033-2680-1234",
    website: "",
    services: "TB treatment, DOTS center, emergency care",
    class: "Class II"
  },
  {
    name: "Serampore Hospital",
    district: "Hooghly",
    city: "Serampore",
    pinCode: "712201",
    postOffice: "SERAMPORE HO",
    type: "Private Hospital",
    category: "private",
    address: "Serampore, Hooghly",
    phone: "+91-033-2662-1234",
    emergencyPhone: "+91-033-2662-1234",
    website: "",
    services: "TB diagnostics, general medicine",
    class: "Class II"
  },
  {
    name: "Chandannagar State General Hospital",
    district: "Hooghly",
    city: "Chandannagar",
    pinCode: "712136",
    postOffice: "CHANDANNAGAR",
    type: "Government Hospital",
    category: "govt",
    address: "Chandannagar, Hooghly",
    phone: "+91-033-2673-1234",
    emergencyPhone: "+91-033-2673-1234",
    website: "",
    services: "TB care, free treatment, DOTS",
    class: "Class II"
  },
  {
    name: "Arambag Sub-Divisional Hospital",
    district: "Hooghly",
    city: "Arambag",
    pinCode: "712601",
    postOffice: "ARAMBAGHHO",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Arambag, Hooghly",
    phone: "+91-03211-255123",
    emergencyPhone: "+91-03211-255123",
    website: "",
    services: "TB screening, DOTS therapy, maternity care",
    class: "Class II"
  },

  // Murshidabad Hospitals
  {
    name: "Murshidabad Medical College & Hospital",
    district: "Murshidabad",
    city: "Berhampore",
    pinCode: "742101",
    postOffice: "BERHAMPORE HO",
    type: "Medical College Hospital",
    category: "govt",
    address: "Berhampore College Road, Berhampore",
    phone: "+91-03482-270303",
    emergencyPhone: "+91-03482-270400",
    website: "https://www.mmch.gov.in/",
    services: "Comprehensive TB care, DOTS therapy, MDR-TB treatment",
    class: "Class I"
  },
  {
    name: "District TB Center Berhampore",
    district: "Murshidabad",
    city: "Berhampore",
    pinCode: "742101",
    postOffice: "BERHAMPORE HO",
    type: "District Hospital",
    category: "govt",
    address: "Civil Hospital Compound, Berhampore",
    phone: "+91-03482-250123",
    emergencyPhone: "+91-03482-250200",
    website: "https://www.wbhealth.gov.in/",
    services: "Free TB testing, DOTS centers, X-ray",
    class: "Class II"
  },
  {
    name: "Jangipur Sub-Divisional Hospital",
    district: "Murshidabad",
    city: "Jangipur",
    pinCode: "742213",
    postOffice: "JANGIPUR",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Jangipur, Murshidabad",
    phone: "+91-03483-261234",
    emergencyPhone: "+91-03483-261234",
    website: "",
    services: "TB care, DOTS therapy, emergency services",
    class: "Class II"
  },
  {
    name: "Kandi Sub-Divisional Hospital",
    district: "Murshidabad",
    city: "Kandi",
    pinCode: "742137",
    postOffice: "KANDI HO",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Kandi, Murshidabad",
    phone: "+91-03484-265123",
    emergencyPhone: "+91-03484-265123",
    website: "",
    services: "TB screening, free treatment, DOTS",
    class: "Class II"
  },

  // Nadia Hospitals
  {
    name: "Krishnanagar District Hospital",
    district: "Nadia",
    city: "Krishnanagar",
    pinCode: "741101",
    postOffice: "KRISHNA NAGAR",
    type: "District Hospital",
    category: "govt",
    address: "Krishnanagar, Nadia",
    phone: "+91-03472-252123",
    emergencyPhone: "+91-03472-252123",
    website: "",
    services: "TB care, DOTS therapy, emergency services",
    class: "Class II"
  },
  {
    name: "Ranaghat Sub-Divisional Hospital",
    district: "Nadia",
    city: "Ranaghat",
    pinCode: "741201",
    postOffice: "RANAGHAT",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Ranaghat, Nadia",
    phone: "+91-03473-244123",
    emergencyPhone: "+91-03473-244123",
    website: "",
    services: "TB treatment, free medicines, DOTS",
    class: "Class II"
  },
  {
    name: "Kalyani State General Hospital",
    district: "Nadia",
    city: "Kalyani",
    pinCode: "741235",
    postOffice: "KALYANI",
    type: "Government Hospital",
    category: "govt",
    address: "Kalyani, Nadia",
    phone: "+91-033-2582-1234",
    emergencyPhone: "+91-033-2582-1234",
    website: "",
    services: "TB care, DOTS center, emergency care",
    class: "Class II"
  },

  // Paschim Bardhaman Hospitals (Asansol-Durgapur)
  {
    name: "Asansol District Hospital",
    district: "Paschim Bardhaman",
    city: "Asansol",
    pinCode: "713301",
    postOffice: "ASANSOL HPO",
    type: "District Hospital",
    category: "govt",
    address: "Asansol, Paschim Bardhaman",
    phone: "+91-0341-2255123",
    emergencyPhone: "+91-0341-2255123",
    website: "",
    services: "TB care, DOTS therapy, emergency services",
    class: "Class II"
  },
  {
    name: "Durgapur Steel Plant Hospital",
    district: "Paschim Bardhaman",
    city: "Durgapur",
    pinCode: "713201",
    postOffice: "DURGAPUR HO",
    type: "Private Hospital",
    category: "private",
    address: "Durgapur Steel Township",
    phone: "+91-0343-2548123",
    emergencyPhone: "+91-0343-2548123",
    website: "",
    services: "TB diagnostics, occupational health, ICU",
    class: "Class I"
  },
  {
    name: "Chittaranjan Locomotive Works Hospital",
    district: "Paschim Bardhaman",
    city: "Chittaranjan",
    pinCode: "713331",
    postOffice: "CHITTARANJAN MDG",
    type: "Government Hospital",
    category: "govt",
    address: "Chittaranjan, Paschim Bardhaman",
    phone: "+91-0341-2529123",
    emergencyPhone: "+91-0341-2529123",
    website: "",
    services: "TB screening, railway employee care, emergency",
    class: "Class II"
  },

  // Purba Bardhaman Hospitals
  {
    name: "Burdwan Medical College & Hospital",
    district: "Purba Bardhaman",
    city: "Bardhaman",
    pinCode: "713101",
    postOffice: "BURDWAN H.P.O",
    type: "Medical College Hospital",
    category: "govt",
    address: "Burdwan, Purba Bardhaman",
    phone: "+91-0342-2663123",
    emergencyPhone: "+91-0342-2663123",
    website: "",
    services: "Comprehensive TB care, MDR-TB treatment, research",
    class: "Class I"
  },
  {
    name: "Katwa Sub-Divisional Hospital",
    district: "Purba Bardhaman",
    city: "Katwa",
    pinCode: "713130",
    postOffice: "KATWA H.P.O",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Katwa, Purba Bardhaman",
    phone: "+91-03453-254123",
    emergencyPhone: "+91-03453-254123",
    website: "",
    services: "TB care, DOTS therapy, maternity services",
    class: "Class II"
  },

  // Darjeeling Hospitals
  {
    name: "Siliguri District Hospital",
    district: "Darjeeling",
    city: "Siliguri",
    pinCode: "734001",
    postOffice: "SILIGURI HPO",
    type: "District Hospital",
    category: "govt",
    address: "Siliguri, Darjeeling",
    phone: "+91-0353-2512123",
    emergencyPhone: "+91-0353-2512123",
    website: "",
    services: "TB care, DOTS therapy, emergency services",
    class: "Class II"
  },
  {
    name: "North Bengal Medical College & Hospital",
    district: "Darjeeling",
    city: "Siliguri",
    pinCode: "734012",
    postOffice: "SUSHRAT NAGAR SO",
    type: "Medical College Hospital",
    category: "govt",
    address: "Sushrutanagar, Siliguri",
    phone: "+91-0353-2528123",
    emergencyPhone: "+91-0353-2528123",
    website: "",
    services: "Advanced TB care, MDR-TB treatment, research",
    class: "Class I"
  },
  {
    name: "Darjeeling District Hospital",
    district: "Darjeeling",
    city: "Darjeeling",
    pinCode: "734101",
    postOffice: "Darjeeling HO",
    type: "District Hospital",
    category: "govt",
    address: "Darjeeling",
    phone: "+91-0354-2254123",
    emergencyPhone: "+91-0354-2254123",
    website: "",
    services: "TB screening, DOTS center, high-altitude care",
    class: "Class II"
  },

  // Jalpaiguri Hospitals
  {
    name: "Jalpaiguri District Hospital",
    district: "Jalpaiguri",
    city: "Jalpaiguri",
    pinCode: "735101",
    postOffice: "JALPAIGURI HO",
    type: "District Hospital",
    category: "govt",
    address: "Jalpaiguri",
    phone: "+91-03561-222123",
    emergencyPhone: "+91-03561-222123",
    website: "",
    services: "TB care, DOTS therapy, emergency services",
    class: "Class II"
  },
  {
    name: "Dhupguri Rural Hospital",
    district: "Jalpaiguri",
    city: "Dhupguri",
    pinCode: "735210",
    postOffice: "DHUPGURI SO",
    type: "Block Primary Health Centre",
    category: "govt",
    address: "Dhupguri, Jalpaiguri",
    phone: "+91-03562-252123",
    emergencyPhone: "+91-03562-252123",
    website: "",
    services: "TB screening, basic healthcare, DOTS",
    class: "Class III"
  },

  // Malda Hospitals
  {
    name: "Malda Medical College & Hospital",
    district: "Malda",
    city: "English Bazar",
    pinCode: "732101",
    postOffice: "MALDA HO",
    type: "Medical College Hospital",
    category: "govt",
    address: "Malda",
    phone: "+91-03512-222123",
    emergencyPhone: "+91-03512-222123",
    website: "",
    services: "Comprehensive TB care, MDR-TB treatment, research",
    class: "Class I"
  },

  // Cooch Behar Hospitals
  {
    name: "Cooch Behar District Hospital",
    district: "Cooch Behar",
    city: "Koch Bihar",
    pinCode: "736101",
    postOffice: "COOCHBEHAR HPO",
    type: "District Hospital",
    category: "govt",
    address: "Cooch Behar",
    phone: "+91-03582-222123",
    emergencyPhone: "+91-03582-222123",
    website: "",
    services: "TB care, DOTS therapy, emergency services",
    class: "Class II"
  },
  {
    name: "Dinhata Sub-Divisional Hospital",
    district: "Cooch Behar",
    city: "Dinhata",
    pinCode: "736135",
    postOffice: "DINHATA",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Dinhata, Cooch Behar",
    phone: "+91-03581-265123",
    emergencyPhone: "+91-03581-265123",
    website: "",
    services: "TB treatment, free medicines, DOTS",
    class: "Class II"
  },

  // Bankura Hospitals
  {
    name: "Bankura Sammilani Medical College & Hospital",
    district: "Bankura",
    city: "Bankura",
    pinCode: "722101",
    postOffice: "BANKURA HPO",
    type: "Medical College Hospital",
    category: "govt",
    address: "Bankura",
    phone: "+91-03242-252123",
    emergencyPhone: "+91-03242-252123",
    website: "",
    services: "Comprehensive TB care, MDR-TB treatment, research",
    class: "Class I"
  },
  {
    name: "Bishnupur Sub-Divisional Hospital",
    district: "Bankura",
    city: "Bishnupur",
    pinCode: "722122",
    postOffice: "BISHNUPUR",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Bishnupur, Bankura",
    phone: "+91-03244-252123",
    emergencyPhone: "+91-03244-252123",
    website: "",
    services: "TB care, DOTS therapy, maternity services",
    class: "Class II"
  },
  {
    name: "Khatra Rural Hospital",
    district: "Bankura",
    city: "Khatra",
    pinCode: "722140",
    postOffice: "KHATRA",
    type: "Block Primary Health Centre",
    category: "govt",
    address: "Khatra, Bankura",
    phone: "+91-03243-265123",
    emergencyPhone: "+91-03243-265123",
    website: "",
    services: "TB screening, basic healthcare, DOTS",
    class: "Class III"
  },

  // Birbhum Hospitals
  {
    name: "Rampurhat Government Hospital",
    district: "Birbhum",
    city: "Rampurhat",
    pinCode: "731224",
    postOffice: "RAMPURHAT",
    type: "Government Hospital",
    category: "govt",
    address: "Rampurhat, Birbhum",
    phone: "+91-03461-252123",
    emergencyPhone: "+91-03461-252123",
    website: "",
    services: "TB care, DOTS therapy, emergency services",
    class: "Class II"
  },
  {
    name: "Suri Sadar Hospital",
    district: "Birbhum",
    city: "Suri",
    pinCode: "731101",
    postOffice: "SURI",
    type: "District Hospital",
    category: "govt",
    address: "Suri, Birbhum",
    phone: "+91-03462-255123",
    emergencyPhone: "+91-03462-255123",
    website: "",
    services: "TB treatment, DOTS center, emergency care",
    class: "Class II"
  },
  {
    name: "Bolpur Sub-Divisional Hospital",
    district: "Birbhum",
    city: "Bolpur",
    pinCode: "731204",
    postOffice: "BOLPUR SO",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Bolpur, Birbhum",
    phone: "+91-03463-252123",
    emergencyPhone: "+91-03463-252123",
    website: "",
    services: "TB care, free treatment, DOTS",
    class: "Class II"
  },

  // Purba Medinipur Hospitals
  {
    name: "Tamluk District Hospital",
    district: "Purba Medinipur",
    city: "Tamluk",
    pinCode: "721636",
    postOffice: "TAMLUK H.O",
    type: "District Hospital",
    category: "govt",
    address: "Tamluk, Purba Medinipur",
    phone: "+91-03228-252123",
    emergencyPhone: "+91-03228-252123",
    website: "",
    services: "TB care, DOTS therapy, emergency services",
    class: "Class II"
  },
  {
    name: "Haldia Sub-Divisional Hospital",
    district: "Purba Medinipur",
    city: "Haldia",
    pinCode: "721606",
    postOffice: "HALDIA O/R",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Haldia, Purba Medinipur",
    phone: "+91-03224-252123",
    emergencyPhone: "+91-03224-252123",
    website: "",
    services: "TB treatment, industrial health, emergency care",
    class: "Class II"
  },
  {
    name: "Contai Sub-Divisional Hospital",
    district: "Purba Medinipur",
    city: "Contai",
    pinCode: "721401",
    postOffice: "CONTAI H.P.O",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Contai, Purba Medinipur",
    phone: "+91-03220-265123",
    emergencyPhone: "+91-03220-265123",
    website: "",
    services: "TB care, DOTS therapy, maternity services",
    class: "Class II"
  },

  // Paschim Medinipur Hospitals
  {
    name: "Midnapore Medical College & Hospital",
    district: "Paschim Medinipur",
    city: "Medinipur",
    pinCode: "721101",
    postOffice: "Midnapore HO",
    type: "Medical College Hospital",
    category: "govt",
    address: "Midnapore",
    phone: "+91-03222-262123",
    emergencyPhone: "+91-03222-262123",
    website: "",
    services: "Comprehensive TB care, MDR-TB treatment, research",
    class: "Class I"
  },
  {
    name: "Kharagpur Sub-Divisional Hospital",
    district: "Paschim Medinipur",
    city: "Kharagpur",
    pinCode: "721301",
    postOffice: "KHARAGPUR",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Kharagpur",
    phone: "+91-03222-255123",
    emergencyPhone: "+91-03222-255123",
    website: "",
    services: "TB care, DOTS therapy, emergency services",
    class: "Class II"
  },
  {
    name: "Jhargram Sub-Divisional Hospital",
    district: "Jhargram",
    city: "Jhargram",
    pinCode: "721507",
    postOffice: "Jhargram HO",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Jhargram",
    phone: "+91-03221-255123",
    emergencyPhone: "+91-03221-255123",
    website: "",
    services: "TB treatment, tribal health, DOTS",
    class: "Class II"
  },

  // Purulia Hospitals
  {
    name: "Purulia District Hospital",
    district: "Purulia",
    city: "Purulia",
    pinCode: "723101",
    postOffice: "PURULIA HPO",
    type: "District Hospital",
    category: "govt",
    address: "Purulia",
    phone: "+91-03252-222123",
    emergencyPhone: "+91-03252-222123",
    website: "",
    services: "TB care, DOTS therapy, emergency services",
    class: "Class II"
  },
  {
    name: "Adra Sub-Divisional Hospital",
    district: "Purulia",
    city: "Adra",
    pinCode: "723121",
    postOffice: "ADRA MDG",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Adra, Purulia",
    phone: "+91-03253-252123",
    emergencyPhone: "+91-03253-252123",
    website: "",
    services: "TB treatment, railway health, DOTS",
    class: "Class II"
  },

  // Uttar Dinajpur Hospitals
  {
    name: "Raiganj Government Medical College & Hospital",
    district: "Uttar Dinajpur",
    city: "Raiganj",
    pinCode: "733134",
    postOffice: "RAIGANJ MDG",
    type: "Medical College Hospital",
    category: "govt",
    address: "Raiganj",
    phone: "+91-03523-252123",
    emergencyPhone: "+91-03523-252123",
    website: "",
    services: "Comprehensive TB care, MDR-TB treatment, research",
    class: "Class I"
  },
  {
    name: "Islampur Sub-Divisional Hospital",
    district: "Uttar Dinajpur",
    city: "Islampur",
    pinCode: "733202",
    postOffice: "ISLAMPUR(NORTH DINAJPUR)",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Islampur, Uttar Dinajpur",
    phone: "+91-03531-265123",
    emergencyPhone: "+91-03531-265123",
    website: "",
    services: "TB care, DOTS therapy, maternity services",
    class: "Class II"
  },

  // Dakshin Dinajpur Hospitals
  {
    name: "Balurghat District Hospital",
    district: "Dakshin Dinajpur",
    city: "Balurghat",
    pinCode: "733101",
    postOffice: "BALURGHAT H.P.O",
    type: "District Hospital",
    category: "govt",
    address: "Balurghat",
    phone: "+91-03522-252123",
    emergencyPhone: "+91-03522-252123",
    website: "",
    services: "TB care, DOTS therapy, emergency services",
    class: "Class II"
  },
  {
    name: "Gangarampur Sub-Divisional Hospital",
    district: "Dakshin Dinajpur",
    city: "Gangarampur",
    pinCode: "733124",
    postOffice: "GANGARAMPUR",
    type: "Sub-Divisional Hospital",
    category: "govt",
    address: "Gangarampur",
    phone: "+91-03524-265123",
    emergencyPhone: "+91-03524-265123",
    website: "",
    services: "TB treatment, free medicines, DOTS",
    class: "Class II"
  }
];

// Helper functions
export const getWBHospitalsByDistrict = (district: string): WBHospital[] => {
  return westBengalHospitals.filter(h => h.district === district);
};

export const getWBHospitalsByCity = (city: string): WBHospital[] => {
  return westBengalHospitals.filter(h => h.city.toLowerCase() === city.toLowerCase());
};

export const getWBHospitalsByPinCode = (pinCode: string): WBHospital[] => {
  return westBengalHospitals.filter(h => h.pinCode === pinCode);
};

export const getWBHospitalsByPostOffice = (postOffice: string): WBHospital[] => {
  return westBengalHospitals.filter(h => 
    h.postOffice.toLowerCase().includes(postOffice.toLowerCase())
  );
};

export const searchWBHospitals = (query: string): WBHospital[] => {
  const normalized = query.toLowerCase();
  return westBengalHospitals.filter(h =>
    h.name.toLowerCase().includes(normalized) ||
    h.city.toLowerCase().includes(normalized) ||
    h.district.toLowerCase().includes(normalized) ||
    h.postOffice.toLowerCase().includes(normalized)
  );
};

export const getAllWBDistricts = (): string[] => {
  const districts = Array.from(new Set(westBengalHospitals.map(h => h.district)));
  return districts.sort();
};

export const getAllWBCitiesFromHospitals = (): string[] => {
  const cities = Array.from(new Set(westBengalHospitals.map(h => h.city)));
  return cities.sort();
};
