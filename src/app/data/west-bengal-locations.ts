// Comprehensive West Bengal Districts, Municipalities, Urban Areas & Post Offices

export interface WBUrbanArea {
  name: string;
  type: 'Municipality' | 'Municipal Corporation' | 'Notified Area' | 'Urban Area';
}

export interface WBPostOffice {
  pinCode: string;
  name: string;
  division: string;
  status: 'Active';
}

export interface WBDistrict {
  name: string;
  urbanAreas: WBUrbanArea[];
  postOffices: WBPostOffice[];
}

export const westBengalDistricts: WBDistrict[] = [
  {
    name: "Alipurduar",
    urbanAreas: [
      { name: "Alipurduar", type: "Municipality" },
      { name: "Falakata", type: "Municipality" },
      { name: "Alipurduar Railway Junction", type: "Urban Area" },
      { name: "Bholar Dabri", type: "Urban Area" },
      { name: "Chechakhata", type: "Urban Area" },
      { name: "Jaygaon", type: "Urban Area" },
      { name: "Paschim Jitpur", type: "Urban Area" },
      { name: "Sobhaganj", type: "Urban Area" },
      { name: "Uttar Kamakhyaguri", type: "Urban Area" },
      { name: "Uttar Latabari", type: "Urban Area" }
    ],
    postOffices: [
      { pinCode: "736121", name: "ALIPURDUAR MDG", division: "COOCH BEHAR", status: "Active" },
      { pinCode: "736123", name: "ALIPURDUAR JUNCTION", division: "COOCH BEHAR", status: "Active" }
    ]
  },
  {
    name: "Bankura",
    urbanAreas: [
      { name: "Bankura", type: "Municipality" },
      { name: "Bishnupur", type: "Municipality" },
      { name: "Sonamukhi", type: "Municipality" },
      { name: "Barjora", type: "Urban Area" },
      { name: "Beliatore", type: "Urban Area" },
      { name: "Khatra", type: "Urban Area" }
    ],
    postOffices: [
      { pinCode: "722101", name: "BANKURA HPO", division: "BANKURA", status: "Active" },
      { pinCode: "722202", name: "BARJORA", division: "BANKURA", status: "Active" },
      { pinCode: "722122", name: "BISHNUPUR", division: "BANKURA", status: "Active" },
      { pinCode: "722132", name: "CHHATNA", division: "BANKURA", status: "Active" },
      { pinCode: "722205", name: "INDAS", division: "BANKURA", status: "Active" },
      { pinCode: "722143", name: "MEJIA", division: "BANKURA", status: "Active" },
      { pinCode: "722144", name: "ONDA", division: "BANKURA", status: "Active" },
      { pinCode: "722148", name: "RANIBANDH", division: "BANKURA", status: "Active" },
      { pinCode: "722151", name: "SIMLAPAL", division: "BANKURA", status: "Active" },
      { pinCode: "722207", name: "SONAMUKHI", division: "BANKURA", status: "Active" },
      { pinCode: "722201", name: "Akui", division: "BANKURA", status: "Active" },
      { pinCode: "722203", name: "Beliatore", division: "BANKURA", status: "Active" },
      { pinCode: "722140", name: "KHATRA", division: "BANKURA", status: "Active" }
    ]
  },
  {
    name: "Birbhum",
    urbanAreas: [
      { name: "Bolpur", type: "Municipality" },
      { name: "Dubrajpur", type: "Municipality" },
      { name: "Nalhati", type: "Municipality" },
      { name: "Rampurhat", type: "Municipality" },
      { name: "Sainthia", type: "Municipality" },
      { name: "Suri", type: "Municipality" },
      { name: "Ahmadpur", type: "Urban Area" }
    ],
    postOffices: [
      { pinCode: "731236", name: "Sriniketan", division: "BIRBHUM", status: "Active" },
      { pinCode: "731204", name: "BOLPUR SO", division: "BIRBHUM", status: "Active" },
      { pinCode: "731123", name: "DUBRAJPUR", division: "BIRBHUM", status: "Active" },
      { pinCode: "731214", name: "ILLAMBAZER SO", division: "BIRBHUM", status: "Active" },
      { pinCode: "731302", name: "KIRNAHAR SO", division: "BIRBHUM", status: "Active" },
      { pinCode: "731234", name: "Sainthia", division: "BIRBHUM", status: "Active" },
      { pinCode: "731220", name: "NALHATISO", division: "BIRBHUM", status: "Active" },
      { pinCode: "731224", name: "RAMPURHAT", division: "BIRBHUM", status: "Active" },
      { pinCode: "731235", name: "SANTINIKETAN SO", division: "BIRBHUM", status: "Active" },
      { pinCode: "731101", name: "SURI", division: "BIRBHUM", status: "Active" }
    ]
  },
  {
    name: "Cooch Behar",
    urbanAreas: [
      { name: "Dinhata", type: "Municipality" },
      { name: "Koch Bihar", type: "Municipality" },
      { name: "Haldibari", type: "Municipality" },
      { name: "Mathabhanga", type: "Municipality" },
      { name: "Mekliganj", type: "Municipality" },
      { name: "Tufanganj", type: "Municipality" },
      { name: "Bhangri Pratham Khanda", type: "Urban Area" },
      { name: "Guriahati", type: "Urban Area" },
      { name: "Khagrabari", type: "Urban Area" },
      { name: "Kharimala Khagrabari", type: "Urban Area" }
    ],
    postOffices: [
      { pinCode: "736101", name: "COOCHBEHAR HPO", division: "COOCH BEHAR", status: "Active" },
      { pinCode: "736135", name: "DINHATA", division: "COOCH BEHAR", status: "Active" },
      { pinCode: "736146", name: "MATHABHANGA SO", division: "COOCH BEHAR", status: "Active" },
      { pinCode: "736159", name: "TUFANGANJ", division: "COOCH BEHAR", status: "Active" },
      { pinCode: "735215", name: "HASIMARA", division: "COOCH BEHAR", status: "Active" }
    ]
  },
  {
    name: "Dakshin Dinajpur",
    urbanAreas: [
      { name: "Balurghat", type: "Municipality" },
      { name: "Gangarampur", type: "Municipality" },
      { name: "Buniadpur", type: "Municipality" }
    ],
    postOffices: [
      { pinCode: "733101", name: "BALURGHAT H.P.O", division: "WEST DINAJPUR", status: "Active" },
      { pinCode: "733121", name: "BUNIADPUR", division: "WEST DINAJPUR", status: "Active" },
      { pinCode: "733124", name: "GANGARAMPUR", division: "WEST DINAJPUR", status: "Active" }
    ]
  },
  {
    name: "Darjeeling",
    urbanAreas: [
      { name: "Darjeeling", type: "Municipality" },
      { name: "Kurseong", type: "Municipality" },
      { name: "Mirik", type: "Notified Area" },
      { name: "Bairatisal", type: "Urban Area" },
      { name: "Cart Road", type: "Urban Area" },
      { name: "Pattabong Tea Garden", type: "Urban Area" },
      { name: "Uttar Bagdogra", type: "Urban Area" }
    ],
    postOffices: [
      { pinCode: "734101", name: "Darjeeling HO", division: "DARJEELING", status: "Active" },
      { pinCode: "734011", name: "KADAMTALA SO", division: "DARJEELING", status: "Active" },
      { pinCode: "734429", name: "NAKSALBARI SO", division: "DARJEELING", status: "Active" },
      { pinCode: "734013", name: "NORTH BENGAL UNIVERSITY SO", division: "DARJEELING", status: "Active" },
      { pinCode: "734003", name: "PRADHAN NAGAR SO", division: "DARJEELING", status: "Active" },
      { pinCode: "734001", name: "SILIGURI HPO", division: "DARJEELING", status: "Active" },
      { pinCode: "734005", name: "SILIGURI BAZAR", division: "DARJEELING", status: "Active" },
      { pinCode: "734012", name: "SUSHRAT NAGAR SO", division: "DARJEELING", status: "Active" }
    ]
  },
  {
    name: "Hooghly",
    urbanAreas: [
      { name: "Chandannagar", type: "Municipal Corporation" },
      { name: "Arambag", type: "Municipality" },
      { name: "Bansberia", type: "Municipality" },
      { name: "Baidyabati", type: "Municipality" },
      { name: "Bhadreswar", type: "Municipality" },
      { name: "Champdani", type: "Municipality" },
      { name: "Hugli-Chuchura", type: "Municipality" },
      { name: "Konnagar", type: "Municipality" },
      { name: "Rishra", type: "Municipality" },
      { name: "Serampore", type: "Municipality" },
      { name: "Tarakeswar", type: "Municipality" },
      { name: "Uttarpara Kotrung", type: "Municipality" }
    ],
    postOffices: [
      { pinCode: "712601", name: "ARAMBAGHHO", division: "HOOGHLY NORTH", status: "Active" },
      { pinCode: "712101", name: "CHINSURAH", division: "HOOGHLY NORTH", status: "Active" },
      { pinCode: "712103", name: "HOOGHLY", division: "HOOGHLY NORTH", status: "Active" },
      { pinCode: "712222", name: "BAIDYABATI", division: "HOOGHLY SOUTH", status: "Active" },
      { pinCode: "712124", name: "BHADRESWAR", division: "HOOGHLY SOUTH", status: "Active" },
      { pinCode: "712136", name: "CHANDANNAGAR", division: "HOOGHLY SOUTH", status: "Active" },
      { pinCode: "712235", name: "KONNAGAR", division: "HOOGHLY SOUTH", status: "Active" },
      { pinCode: "712248", name: "RISHRA", division: "HOOGHLY SOUTH", status: "Active" },
      { pinCode: "712201", name: "SERAMPORE HO", division: "HOOGHLY SOUTH", status: "Active" },
      { pinCode: "712410", name: "TARAKESWAR", division: "HOOGHLY SOUTH", status: "Active" },
      { pinCode: "712258", name: "UTTARPARA MDG", division: "HOOGHLY SOUTH", status: "Active" }
    ]
  },
  {
    name: "Howrah",
    urbanAreas: [
      { name: "Howrah", type: "Municipal Corporation" },
      { name: "Bally", type: "Municipality" },
      { name: "Balichak", type: "Municipality" },
      { name: "Bagnan", type: "Municipality" },
      { name: "Bankra", type: "Municipality" },
      { name: "Beldubi", type: "Municipality" }
    ],
    postOffices: [
      { pinCode: "711401", name: "AMTA PO", division: "HOWRAH", status: "Active" },
      { pinCode: "711201", name: "BALLY PO", division: "HOWRAH", status: "Active" },
      { pinCode: "711303", name: "BAGNAN PO", division: "HOWRAH", status: "Active" },
      { pinCode: "711202", name: "BELUR PO", division: "HOWRAH", status: "Active" },
      { pinCode: "711405", name: "DOMJUR PO", division: "HOWRAH", status: "Active" },
      { pinCode: "711101", name: "HOWRAH HPO", division: "HOWRAH", status: "Active" },
      { pinCode: "711204", name: "LILUAH PO", division: "HOWRAH", status: "Active" },
      { pinCode: "711106", name: "SALKIA HPO", division: "HOWRAH", status: "Active" },
      { pinCode: "711104", name: "SANTRAGACHI", division: "HOWRAH", status: "Active" },
      { pinCode: "711315", name: "ULUBERIA", division: "HOWRAH", status: "Active" }
    ]
  },
  {
    name: "Jalpaiguri",
    urbanAreas: [
      { name: "Jalpaiguri", type: "Municipality" },
      { name: "Dhupguri", type: "Municipality" },
      { name: "Mainaguri", type: "Municipality" }
    ],
    postOffices: [
      { pinCode: "735101", name: "JALPAIGURI HO", division: "JALPAIGURI", status: "Active" },
      { pinCode: "735210", name: "DHUPGURI SO", division: "JALPAIGURI", status: "Active" },
      { pinCode: "735224", name: "MOYNAGURI SO", division: "JALPAIGURI", status: "Active" },
      { pinCode: "735204", name: "BIRPARA", division: "JALPAIGURI", status: "Active" },
      { pinCode: "735221", name: "MAL HPO", division: "JALPAIGURI", status: "Active" }
    ]
  },
  {
    name: "Kolkata",
    urbanAreas: [
      { name: "Kolkata", type: "Municipal Corporation" }
    ],
    postOffices: [
      { pinCode: "700001", name: "Kolkata GPO", division: "Self-Unit", status: "Active" },
      { pinCode: "700007", name: "Barabazar H.O.", division: "Self-Unit", status: "Active" },
      { pinCode: "700027", name: "Alipore H.O.", division: "Self-Unit", status: "Active" },
      { pinCode: "700010", name: "Beleghata H.O.", division: "East Kolkata", status: "Active" },
      { pinCode: "700017", name: "Circus avenue", division: "East Kolkata", status: "Active" },
      { pinCode: "700004", name: "Shyambazar", division: "North Kolkata", status: "Active" },
      { pinCode: "700029", name: "Sarat Bose Road", division: "South Kolkata", status: "Active" },
      { pinCode: "700019", name: "Ballygunge MDG", division: "South Kolkata", status: "Active" },
      { pinCode: "700012", name: "Bowbazar", division: "Central Kolkata", status: "Active" }
    ]
  },
  {
    name: "Malda",
    urbanAreas: [
      { name: "English Bazar", type: "Municipality" },
      { name: "Old Malda", type: "Municipality" }
    ],
    postOffices: [
      { pinCode: "732101", name: "MALDA HO", division: "MALDA", status: "Active" },
      { pinCode: "732210", name: "BAISHNABNAGAR", division: "MALDA", status: "Active" },
      { pinCode: "732123", name: "CHANCHAL SUB POST OFFICE", division: "MALDA", status: "Active" },
      { pinCode: "732124", name: "GAJOL", division: "MALDA", status: "Active" }
    ]
  },
  {
    name: "Murshidabad",
    urbanAreas: [
      { name: "Baharampur", type: "Municipality" },
      { name: "Dhulian", type: "Municipality" },
      { name: "Jangipur", type: "Municipality" },
      { name: "Jiaganj-Azimganj", type: "Municipality" },
      { name: "Beldanga", type: "Municipality" },
      { name: "Kandi", type: "Municipality" },
      { name: "Domkal", type: "Municipality" },
      { name: "Murshidabad", type: "Municipality" }
    ],
    postOffices: [
      { pinCode: "742101", name: "BERHAMPORE HO", division: "MURSHIDABAD", status: "Active" },
      { pinCode: "742201", name: "AURANGABAD S.O", division: "MURSHIDABAD", status: "Active" },
      { pinCode: "742133", name: "BELDANGA SO", division: "MURSHIDABAD", status: "Active" },
      { pinCode: "742202", name: "DHULIYAN", division: "MURSHIDABAD", status: "Active" },
      { pinCode: "742213", name: "JANGIPUR", division: "MURSHIDABAD", status: "Active" },
      { pinCode: "742123", name: "JIAGANJ", division: "MURSHIDABAD", status: "Active" },
      { pinCode: "742137", name: "KANDI HO", division: "MURSHIDABAD", status: "Active" },
      { pinCode: "742149", name: "MURSHIDABAD", division: "MURSHIDABAD", status: "Active" }
    ]
  },
  {
    name: "Nadia",
    urbanAreas: [
      { name: "Krishnanagar", type: "Municipality" },
      { name: "Nabadwip", type: "Municipality" },
      { name: "Ranaghat", type: "Municipality" },
      { name: "Santipur", type: "Municipality" },
      { name: "Chakdaha", type: "Municipality" }
    ],
    postOffices: [
      { pinCode: "741101", name: "KRISHNA NAGAR", division: "NADIA NORTH", status: "Active" },
      { pinCode: "741302", name: "NABADWIP", division: "NADIA NORTH", status: "Active" },
      { pinCode: "741160", name: "TEHATTA", division: "NADIA NORTH", status: "Active" },
      { pinCode: "741201", name: "RANAGHAT", division: "NADIA SOUTH", status: "Active" },
      { pinCode: "741404", name: "SANTIPUR", division: "NADIA SOUTH", status: "Active" },
      { pinCode: "741222", name: "CHAKDAHA", division: "NADIA SOUTH", status: "Active" },
      { pinCode: "741235", name: "KALYANI", division: "NADIA SOUTH", status: "Active" }
    ]
  },
  {
    name: "North 24 Parganas",
    urbanAreas: [
      { name: "Barasat", type: "Municipality" },
      { name: "Baranagar", type: "Municipality" },
      { name: "Basirhat", type: "Municipality" },
      { name: "Habra", type: "Municipality" },
      { name: "North Dum Dum", type: "Municipality" },
      { name: "Panihati", type: "Municipality" },
      { name: "Naihati", type: "Municipality" },
      { name: "New Barrackpore", type: "Municipality" },
      { name: "North Barrackpore", type: "Municipality" }
    ],
    postOffices: [
      { pinCode: "700124", name: "BARASAT HO", division: "BARASAT", status: "Active" },
      { pinCode: "743411", name: "BASIRHAT H.P.O.", division: "BARASAT", status: "Active" },
      { pinCode: "743263", name: "HABRA", division: "BARASAT", status: "Active" },
      { pinCode: "700129", name: "MADHYAMGRAM", division: "BARASAT", status: "Active" },
      { pinCode: "700109", name: "Agarpara", division: "North Presidency", status: "Active" },
      { pinCode: "700120", name: "Barrackpore", division: "North Presidency", status: "Active" },
      { pinCode: "743165", name: "Naihati", division: "North Presidency", status: "Active" },
      { pinCode: "743145", name: "Panihati", division: "North Presidency", status: "Active" }
    ]
  },
  {
    name: "Paschim Medinipur",
    urbanAreas: [
      { name: "Kharagpur", type: "Municipality" },
      { name: "Medinipur", type: "Municipality" },
      { name: "Chandrakona", type: "Municipality" },
      { name: "Jhargram", type: "Municipality" }
    ],
    postOffices: [
      { pinCode: "721301", name: "KHARAGPUR", division: "MIDNAPORE", status: "Active" },
      { pinCode: "721101", name: "Midnapore HO", division: "MIDNAPORE", status: "Active" },
      { pinCode: "721201", name: "CHANDRAKONA", division: "MIDNAPORE", status: "Active" },
      { pinCode: "721507", name: "Jhargram HO", division: "MIDNAPORE", status: "Active" }
    ]
  },
  {
    name: "Paschim Bardhaman",
    urbanAreas: [
      { name: "Asansol", type: "Municipal Corporation" },
      { name: "Durgapur", type: "Municipal Corporation" }
    ],
    postOffices: [
      { pinCode: "713301", name: "ASANSOL HPO", division: "ASANSOL", status: "Active" },
      { pinCode: "713201", name: "DURGAPUR HO", division: "ASANSOL", status: "Active" },
      { pinCode: "713324", name: "BARAKAR SO", division: "ASANSOL", status: "Active" },
      { pinCode: "713331", name: "CHITTARANJAN MDG", division: "ASANSOL", status: "Active" }
    ]
  },
  {
    name: "Purba Bardhaman",
    urbanAreas: [
      { name: "Bardhaman", type: "Municipality" }
    ],
    postOffices: [
      { pinCode: "713101", name: "BURDWAN H.P.O", division: "BURDWAN", status: "Active" },
      { pinCode: "713130", name: "KATWA H.P.O", division: "BURDWAN", status: "Active" },
      { pinCode: "713128", name: "GUSHKARA SO", division: "BURDWAN", status: "Active" }
    ]
  },
  {
    name: "Purba Medinipur",
    urbanAreas: [
      { name: "Contai", type: "Municipality" },
      { name: "Egra", type: "Municipality" },
      { name: "Haldia", type: "Municipality" },
      { name: "Kolaghat", type: "Municipality" },
      { name: "Panskura", type: "Municipality" }
    ],
    postOffices: [
      { pinCode: "721401", name: "CONTAI H.P.O", division: "CONTAI", status: "Active" },
      { pinCode: "721429", name: "EGRA", division: "CONTAI", status: "Active" },
      { pinCode: "721606", name: "HALDIA O/R", division: "TAMLUK", status: "Active" },
      { pinCode: "721139", name: "PANSKURA", division: "TAMLUK", status: "Active" },
      { pinCode: "721636", name: "TAMLUK H.O", division: "TAMLUK", status: "Active" }
    ]
  },
  {
    name: "Purulia",
    urbanAreas: [
      { name: "Purulia", type: "Municipality" }
    ],
    postOffices: [
      { pinCode: "723101", name: "PURULIA HPO", division: "PURULIA", status: "Active" },
      { pinCode: "723121", name: "ADRA MDG", division: "PURULIA", status: "Active" },
      { pinCode: "723131", name: "MANBAZAR", division: "PURULIA", status: "Active" },
      { pinCode: "723133", name: "RAGHUNATHPUR", division: "PURULIA", status: "Active" }
    ]
  },
  {
    name: "South 24 Parganas",
    urbanAreas: [
      { name: "Baruipur", type: "Municipality" },
      { name: "Budge Budge", type: "Municipality" },
      { name: "Diamond Harbour", type: "Municipality" },
      { name: "Jaynagar Majilpur", type: "Municipality" },
      { name: "Maheshtala", type: "Municipality" }
    ],
    postOffices: [
      { pinCode: "700144", name: "BARUIPUR HO", division: "KOL SOUTH PRESIDENCY", status: "Active" },
      { pinCode: "743331", name: "DIAMOND HARBOUR", division: "KOL SOUTH PRESIDENCY", status: "Active" },
      { pinCode: "700084", name: "GARIA SO", division: "KOL SOUTH PRESIDENCY", status: "Active" },
      { pinCode: "700150", name: "SONARPUR SO", division: "KOL SOUTH PRESIDENCY", status: "Active" }
    ]
  },
  {
    name: "Uttar Dinajpur",
    urbanAreas: [
      { name: "Dalkhola", type: "Municipality" },
      { name: "Raiganj", type: "Municipality" },
      { name: "Islampur", type: "Municipality" }
    ],
    postOffices: [
      { pinCode: "733201", name: "DALKHOLA", division: "WEST DINAJPUR", status: "Active" },
      { pinCode: "733134", name: "RAIGANJ MDG", division: "WEST DINAJPUR", status: "Active" },
      { pinCode: "733202", name: "ISLAMPUR(NORTH DINAJPUR)", division: "WEST DINAJPUR", status: "Active" }
    ]
  },
  {
    name: "Kalimpong",
    urbanAreas: [
      { name: "Kalimpong", type: "Municipality" }
    ],
    postOffices: []
  },
  {
    name: "Jhargram",
    urbanAreas: [
      { name: "Jhargram", type: "Municipality" }
    ],
    postOffices: [
      { pinCode: "721507", name: "Jhargram HO", division: "MIDNAPORE", status: "Active" },
      { pinCode: "721507", name: "RAGHUNATHPUR(WEST)", division: "MIDNAPORE", status: "Active" }
    ]
  }
];

// Helper functions
export const getAllWBCities = (): string[] => {
  const cities: string[] = [];
  westBengalDistricts.forEach(district => {
    district.urbanAreas.forEach(area => {
      cities.push(area.name);
    });
  });
  return Array.from(new Set(cities)).sort();
};

export const getWBDistrictByCity = (city: string): string | null => {
  for (const district of westBengalDistricts) {
    const found = district.urbanAreas.find(area => 
      area.name.toLowerCase() === city.toLowerCase()
    );
    if (found) return district.name;
  }
  return null;
};

export const getWBPostOfficesByDistrict = (districtName: string): WBPostOffice[] => {
  const district = westBengalDistricts.find(d => d.name === districtName);
  return district ? district.postOffices : [];
};

export const searchWBCityByPrefix = (prefix: string): string[] => {
  const allCities = getAllWBCities();
  if (!prefix.trim()) return allCities;
  return allCities.filter(city => city.toLowerCase().startsWith(prefix.toLowerCase()));
};
