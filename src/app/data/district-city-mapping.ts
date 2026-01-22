// Comprehensive District-City Mapping for India
// This database contains district-wise city/town listings for accurate location-based hospital search

export interface CityTown {
  code: string;
  name: string;
  districtCode: string;
  districtName: string;
  stateCode: string;
  stateName: string;
}

// Helper to get all cities for a specific district
export const getCitiesByDistrict = (districtName: string, stateName?: string): string[] => {
  let filtered = districtCityData.filter(city => 
    city.districtName.toLowerCase() === districtName.toLowerCase()
  );
  
  if (stateName) {
    filtered = filtered.filter(city => 
      city.stateName.toLowerCase() === stateName.toLowerCase()
    );
  }
  
  const cityNames = Array.from(new Set(filtered.map(city => city.name)));
  return cityNames.sort();
};

// Get all cities for a state
export const getCitiesByState = (stateName: string): string[] => {
  const filtered = districtCityData.filter(city => 
    city.stateName.toLowerCase() === stateName.toLowerCase()
  );
  
  const cityNames = Array.from(new Set(filtered.map(city => city.name)));
  return cityNames.sort();
};

// Get district name for a city
export const getDistrictForCity = (cityName: string, stateName?: string): string | null => {
  let city = districtCityData.find(c => c.name.toLowerCase() === cityName.toLowerCase());
  
  if (stateName && city) {
    city = districtCityData.find(c => 
      c.name.toLowerCase() === cityName.toLowerCase() && 
      c.stateName.toLowerCase() === stateName.toLowerCase()
    );
  }
  
  return city ? city.districtName : null;
};

// Comprehensive district-city database
export const districtCityData: CityTown[] = [
  // West Bengal - Paschim Medinipur District (User's specific example)
  { code: "801752", name: "Chandrakona", districtCode: "344", districtName: "Paschim Medinipur", stateCode: "19", stateName: "West Bengal" },
  { code: "801754", name: "Ghatal", districtCode: "344", districtName: "Paschim Medinipur", stateCode: "19", stateName: "West Bengal" },
  { code: "801756", name: "Jhargram", districtCode: "344", districtName: "Paschim Medinipur", stateCode: "19", stateName: "West Bengal" },
  { code: "801757", name: "Kharagpur", districtCode: "344", districtName: "Paschim Medinipur", stateCode: "19", stateName: "West Bengal" },
  { code: "801753", name: "Kharar", districtCode: "344", districtName: "Paschim Medinipur", stateCode: "19", stateName: "West Bengal" },
  { code: "801751", name: "Khirpai", districtCode: "344", districtName: "Paschim Medinipur", stateCode: "19", stateName: "West Bengal" },
  { code: "801755", name: "Medinipur", districtCode: "344", districtName: "Paschim Medinipur", stateCode: "19", stateName: "West Bengal" },
  { code: "801750", name: "Ramjibanpur", districtCode: "344", districtName: "Paschim Medinipur", stateCode: "19", stateName: "West Bengal" },

  // West Bengal - Other Districts
  { code: "801733", name: "Bankura", districtCode: "339", districtName: "Bankura", stateCode: "19", stateName: "West Bengal" },
  { code: "801735", name: "Bishnupur", districtCode: "339", districtName: "Bankura", stateCode: "19", stateName: "West Bengal" },
  { code: "801734", name: "Sonamukhi", districtCode: "339", districtName: "Bankura", stateCode: "19", stateName: "West Bengal" },
  
  { code: "801669", name: "Bolpur", districtCode: "334", districtName: "Birbhum", stateCode: "19", stateName: "West Bengal" },
  { code: "801668", name: "Dubrajpur", districtCode: "334", districtName: "Birbhum", stateCode: "19", stateName: "West Bengal" },
  { code: "801664", name: "Nalhati", districtCode: "334", districtName: "Birbhum", stateCode: "19", stateName: "West Bengal" },
  { code: "801665", name: "Rampurhat", districtCode: "334", districtName: "Birbhum", stateCode: "19", stateName: "West Bengal" },
  { code: "801667", name: "Sainthia", districtCode: "334", districtName: "Birbhum", stateCode: "19", stateName: "West Bengal" },
  { code: "801666", name: "Suri", districtCode: "334", districtName: "Birbhum", stateCode: "19", stateName: "West Bengal" },
  
  { code: "801742", name: "Kolkata", districtCode: "342", districtName: "Kolkata", stateCode: "19", stateName: "West Bengal" },
  
  { code: "801740", name: "Howrah", districtCode: "341", districtName: "Howrah", stateCode: "19", stateName: "West Bengal" },
  { code: "801741", name: "Uluberia", districtCode: "341", districtName: "Howrah", stateCode: "19", stateName: "West Bengal" },
  
  { code: "801722", name: "Chandannagar", districtCode: "338", districtName: "Hooghly", stateCode: "19", stateName: "West Bengal" },
  { code: "801728", name: "Serampore", districtCode: "338", districtName: "Hooghly", stateCode: "19", stateName: "West Bengal" },
  { code: "801721", name: "Hugli-Chinsurah", districtCode: "338", districtName: "Hooghly", stateCode: "19", stateName: "West Bengal" },
  { code: "801724", name: "Arambag", districtCode: "338", districtName: "Hooghly", stateCode: "19", stateName: "West Bengal" },
  { code: "801720", name: "Bansberia", districtCode: "338", districtName: "Hooghly", stateCode: "19", stateName: "West Bengal" },
  
  { code: "801707", name: "Barasat", districtCode: "337", districtName: "North Twenty Four Parganas", stateCode: "19", stateName: "West Bengal" },
  { code: "801702", name: "Barrackpore", districtCode: "337", districtName: "North Twenty Four Parganas", stateCode: "19", stateName: "West Bengal" },
  { code: "801718", name: "Basirhat", districtCode: "337", districtName: "North Twenty Four Parganas", stateCode: "19", stateName: "West Bengal" },
  { code: "801697", name: "Habra", districtCode: "337", districtName: "North Twenty Four Parganas", stateCode: "19", stateName: "West Bengal" },
  
  { code: "801747", name: "Baruipur", districtCode: "343", districtName: "South Twenty Four Parganas", stateCode: "19", stateName: "West Bengal" },
  { code: "801748", name: "Diamond Harbour", districtCode: "343", districtName: "South Twenty Four Parganas", stateCode: "19", stateName: "West Bengal" },
  
  // Tamil Nadu - Major Districts
  { code: "803339", name: "Chennai", districtCode: "603", districtName: "Chennai", stateCode: "33", stateName: "Tamil Nadu" },
  
  { code: "803984", name: "Coimbatore", districtCode: "632", districtName: "Coimbatore", stateCode: "33", stateName: "Tamil Nadu" },
  { code: "803961", name: "Mettupalayam", districtCode: "632", districtName: "Coimbatore", stateCode: "33", stateName: "Tamil Nadu" },
  { code: "804002", name: "Pollachi", districtCode: "632", districtName: "Coimbatore", stateCode: "33", stateName: "Tamil Nadu" },
  { code: "804010", name: "Valparai", districtCode: "632", districtName: "Coimbatore", stateCode: "33", stateName: "Tamil Nadu" },
  
  { code: "803754", name: "Madurai", districtCode: "623", districtName: "Madurai", stateCode: "33", stateName: "Tamil Nadu" },
  { code: "803741", name: "Melur", districtCode: "623", districtName: "Madurai", stateCode: "33", stateName: "Tamil Nadu" },
  
  { code: "803463", name: "Salem", districtCode: "608", districtName: "Salem", stateCode: "33", stateName: "Tamil Nadu" },
  { code: "803473", name: "Attur", districtCode: "608", districtName: "Salem", stateCode: "33", stateName: "Tamil Nadu" },
  
  { code: "803631", name: "Tiruchirappalli", districtCode: "614", districtName: "Tiruchirappalli", stateCode: "33", stateName: "Tamil Nadu" },
  { code: "803629", name: "Lalgudi", districtCode: "614", districtName: "Tiruchirappalli", stateCode: "33", stateName: "Tamil Nadu" },
  
  // Karnataka - Major Districts
  { code: "803162", name: "Bangalore", districtCode: "572", districtName: "Bangalore", stateCode: "29", stateName: "Karnataka" },
  
  { code: "803083", name: "Hubli-Dharwad", districtCode: "562", districtName: "Dharwad", stateCode: "29", stateName: "Karnataka" },
  
  { code: "803194", name: "Mysore", districtCode: "577", districtName: "Mysore", stateCode: "29", stateName: "Karnataka" },
  { code: "803193", name: "Krishnarajanagara", districtCode: "577", districtName: "Mysore", stateCode: "29", stateName: "Karnataka" },
  
  { code: "803181", name: "Mangalore", districtCode: "575", districtName: "Dakshina Kannada", stateCode: "29", stateName: "Karnataka" },
  { code: "803185", name: "Puttur", districtCode: "575", districtName: "Dakshina Kannada", stateCode: "29", stateName: "Karnataka" },
  
  // Kerala - Major Districts
  { code: "803288", name: "Kochi", districtCode: "595", districtName: "Ernakulam", stateCode: "32", stateName: "Kerala" },
  { code: "803286", name: "Aluva", districtCode: "595", districtName: "Ernakulam", stateCode: "32", stateName: "Kerala" },
  
  { code: "803312", name: "Thiruvananthapuram", districtCode: "601", districtName: "Thiruvananthapuram", stateCode: "32", stateName: "Kerala" },
  { code: "803310", name: "Attingal", districtCode: "601", districtName: "Thiruvananthapuram", stateCode: "32", stateName: "Kerala" },
  
  { code: "803267", name: "Kozhikode", districtCode: "591", districtName: "Kozhikode", stateCode: "32", stateName: "Kerala" },
  { code: "803266", name: "Quilandy", districtCode: "591", districtName: "Kozhikode", stateCode: "32", stateName: "Kerala" },
  
  // Andhra Pradesh - Major Districts
  { code: "803009", name: "Anantapur", districtCode: "553", districtName: "Anantapur", stateCode: "28", stateName: "Andhra Pradesh" },
  { code: "803012", name: "Hindupur", districtCode: "553", districtName: "Anantapur", stateCode: "28", stateName: "Andhra Pradesh" },
  
  { code: "802955", name: "Kakinada", districtCode: "545", districtName: "East Godavari", stateCode: "28", stateName: "Andhra Pradesh" },
  { code: "802952", name: "Rajahmundry", districtCode: "545", districtName: "East Godavari", stateCode: "28", stateName: "Andhra Pradesh" },
  
  { code: "802981", name: "Guntur", districtCode: "548", districtName: "Guntur", stateCode: "28", stateName: "Andhra Pradesh" },
  { code: "802982", name: "Tenali", districtCode: "548", districtName: "Guntur", stateCode: "28", stateName: "Andhra Pradesh" },
  
  { code: "802969", name: "Vijayawada", districtCode: "547", districtName: "Krishna", stateCode: "28", stateName: "Andhra Pradesh" },
  { code: "802970", name: "Gudivada", districtCode: "547", districtName: "Krishna", stateCode: "28", stateName: "Andhra Pradesh" },
  
  { code: "802947", name: "Visakhapatnam (GVMC)", districtCode: "544", districtName: "Visakhapatnam", stateCode: "28", stateName: "Andhra Pradesh" },
  
  // Delhi
  { code: "800442", name: "New Delhi (NDMC)", districtCode: "94", districtName: "New Delhi", stateCode: "07", stateName: "NCT of Delhi" },
  { code: "900050", name: "East Delhi MCD", districtCode: "93", districtName: "East Delhi", stateCode: "07", stateName: "NCT of Delhi" },
  { code: "800441", name: "South Delhi MCD", districtCode: "98", districtName: "South Delhi", stateCode: "07", stateName: "NCT of Delhi" },
  { code: "900051", name: "North Delhi MCD", districtCode: "91", districtName: "North Delhi", stateCode: "07", stateName: "NCT of Delhi" },
  
  // Maharashtra - Major Districts
  { code: "802794", name: "Greater Mumbai", districtCode: "519", districtName: "Mumbai", stateCode: "27", stateName: "Maharashtra" },
  
  { code: "802814", name: "Pune", districtCode: "521", districtName: "Pune", stateCode: "27", stateName: "Maharashtra" },
  { code: "802811", name: "Pimpri Chinchwad", districtCode: "521", districtName: "Pune", stateCode: "27", stateName: "Maharashtra" },
  
  { code: "802710", name: "Nagpur", districtCode: "505", districtName: "Nagpur", stateCode: "27", stateName: "Maharashtra" },
  
  { code: "802787", name: "Thane", districtCode: "517", districtName: "Thane", stateCode: "27", stateName: "Maharashtra" },
  { code: "802790", name: "Kalyan-Dombivli", districtCode: "517", districtName: "Thane", stateCode: "27", stateName: "Maharashtra" },
  
  // Uttar Pradesh - Sample
  { code: "800804", name: "Agra", districtCode: "146", districtName: "Agra", stateCode: "09", stateName: "Uttar Pradesh" },
  { code: "800768", name: "Aligarh", districtCode: "143", districtName: "Aligarh", stateCode: "09", stateName: "Uttar Pradesh" },
  { code: "801086", name: "Allahabad", districtCode: "175", districtName: "Allahabad", stateCode: "09", stateName: "Uttar Pradesh" },
  { code: "800866", name: "Bareilly", districtCode: "150", districtName: "Bareilly", stateCode: "09", stateName: "Uttar Pradesh" },
  { code: "800734", name: "Ghaziabad", districtCode: "140", districtName: "Ghaziabad", stateCode: "09", stateName: "Uttar Pradesh" },
  { code: "801005", name: "Kanpur", districtCode: "164", districtName: "Kanpur Nagar", stateCode: "09", stateName: "Uttar Pradesh" },
  { code: "800951", name: "Lucknow", districtCode: "157", districtName: "Lucknow", stateCode: "09", stateName: "Uttar Pradesh" },
  { code: "800682", name: "Moradabad", districtCode: "135", districtName: "Moradabad", stateCode: "09", stateName: "Uttar Pradesh" },
  { code: "801235", name: "Varanasi", districtCode: "197", districtName: "Varanasi", stateCode: "09", stateName: "Uttar Pradesh" },
  
  // Telangana
  { code: "802918", name: "Hyderabad (GHMC)", districtCode: "536", districtName: "Hyderabad", stateCode: "36", stateName: "Telangana" },
  { code: "802911", name: "Karimnagar", districtCode: "534", districtName: "Karimnagar", stateCode: "36", stateName: "Telangana" },
  { code: "802937", name: "Khammam", districtCode: "541", districtName: "Khammam", stateCode: "36", stateName: "Telangana" },
  { code: "802930", name: "Warangal", districtCode: "540", districtName: "Warangal", stateCode: "36", stateName: "Telangana" },
  
  // Rajasthan
  { code: "800570", name: "Ajmer", districtCode: "119", districtName: "Ajmer", stateCode: "08", stateName: "Rajasthan" },
  { code: "800490", name: "Alwar", districtCode: "104", districtName: "Alwar", stateCode: "08", stateName: "Rajasthan" },
  { code: "800544", name: "Jodhpur", districtCode: "113", districtName: "Jodhpur", stateCode: "08", stateName: "Rajasthan" },
  { code: "800522", name: "Jaipur", districtCode: "110", districtName: "Jaipur", stateCode: "08", stateName: "Rajasthan" },
  { code: "800623", name: "Udaipur", districtCode: "130", districtName: "Udaipur", stateCode: "08", stateName: "Rajasthan" },
  
  // Gujarat
  { code: "802484", name: "Ahmedabad", districtCode: "474", districtName: "Ahmedabad", stateCode: "24", stateName: "Gujarat" },
  { code: "802629", name: "Surat", districtCode: "492", districtName: "Surat", stateCode: "24", stateName: "Gujarat" },
  { code: "802596", name: "Vadodara", districtCode: "486", districtName: "Vadodara", stateCode: "24", stateName: "Gujarat" },
  { code: "802501", name: "Rajkot", districtCode: "476", districtName: "Rajkot", stateCode: "24", stateName: "Gujarat" },
  
  // Punjab
  { code: "800252", name: "Amritsar", districtCode: "49", districtName: "Amritsar", stateCode: "03", stateName: "Punjab" },
  { code: "800226", name: "Bathinda", districtCode: "46", districtName: "Bathinda", stateCode: "03", stateName: "Punjab" },
  { code: "800196", name: "Ludhiana", districtCode: "41", districtName: "Ludhiana", stateCode: "03", stateName: "Punjab" },
  { code: "800166", name: "Jalandhar", districtCode: "37", districtName: "Jalandhar", stateCode: "03", stateName: "Punjab" },
  
  // Haryana
  { code: "800429", name: "Gurgaon", districtCode: "86", districtName: "Gurgaon", stateCode: "06", stateName: "Haryana" },
  { code: "800436", name: "Faridabad", districtCode: "88", districtName: "Faridabad", stateCode: "06", stateName: "Haryana" },
  { code: "800363", name: "Panchkula", districtCode: "69", districtName: "Panchkula", stateCode: "06", stateName: "Haryana" },
  
  // Madhya Pradesh
  { code: "802312", name: "Bhopal", districtCode: "444", districtName: "Bhopal", stateCode: "23", stateName: "Madhya Pradesh" },
  { code: "802273", name: "Indore", districtCode: "439", districtName: "Indore", stateCode: "23", stateName: "Madhya Pradesh" },
  { code: "802361", name: "Jabalpur", districtCode: "451", districtName: "Jabalpur", stateCode: "23", stateName: "Madhya Pradesh" },
  { code: "802100", name: "Gwalior", districtCode: "421", districtName: "Gwalior", stateCode: "23", stateName: "Madhya Pradesh" },
];

// Get all unique districts
export const getAllDistricts = (): Array<{name: string, state: string}> => {
  const districts = new Map<string, {name: string, state: string}>();
  
  districtCityData.forEach(city => {
    const key = `${city.districtName}-${city.stateName}`;
    if (!districts.has(key)) {
      districts.set(key, {
        name: city.districtName,
        state: city.stateName
      });
    }
  });
  
  return Array.from(districts.values()).sort((a, b) => 
    a.name.localeCompare(b.name)
  );
};
