// Comprehensive Indian States, Districts, and Cities/Villages Database

export interface City {
  name: string;
}

export interface District {
  name: string;
  cities: City[];
}

export interface State {
  name: string;
  districts: District[];
}

export const indianStates: State[] = [
  {
    name: "West Bengal",
    districts: [
      {
        name: "Kolkata",
        cities: [
          { name: "Alipore" },
          { name: "Ballygunge" },
          { name: "Behala" },
          { name: "Bhowanipore" },
          { name: "Jadavpur" },
          { name: "Kalighat" },
          { name: "New Alipore" },
          { name: "Salt Lake" },
          { name: "Tollygunge" },
        ]
      },
      {
        name: "Murshidabad",
        cities: [
          { name: "Berhampore" },
          { name: "Kandi" },
          { name: "Jiaganj" },
          { name: "Lalbagh" },
          { name: "Murshidabad" },
          { name: "Nabagram" },
          { name: "Rejinagar" },
        ]
      },
      {
        name: "North 24 Parganas",
        cities: [
          { name: "Barasat" },
          { name: "Barrackpore" },
          { name: "Basirhat" },
          { name: "Bidhannagar" },
          { name: "Dum Dum" },
          { name: "Habra" },
          { name: "Madhyamgram" },
        ]
      },
      {
        name: "South 24 Parganas",
        cities: [
          { name: "Alipore" },
          { name: "Baruipur" },
          { name: "Canning" },
          { name: "Diamond Harbour" },
          { name: "Kakdwip" },
          { name: "Sonarpur" },
        ]
      },
      {
        name: "Howrah",
        cities: [
          { name: "Bally" },
          { name: "Howrah" },
          { name: "Liluah" },
          { name: "Sankrail" },
          { name: "Uluberia" },
        ]
      },
    ]
  },
  {
    name: "Maharashtra",
    districts: [
      {
        name: "Mumbai",
        cities: [
          { name: "Andheri" },
          { name: "Bandra" },
          { name: "Borivali" },
          { name: "Colaba" },
          { name: "Dadar" },
          { name: "Juhu" },
          { name: "Kurla" },
          { name: "Malad" },
          { name: "Powai" },
          { name: "Vile Parle" },
        ]
      },
      {
        name: "Pune",
        cities: [
          { name: "Aundh" },
          { name: "Hadapsar" },
          { name: "Hinjewadi" },
          { name: "Kothrud" },
          { name: "Pimpri" },
          { name: "Shivajinagar" },
          { name: "Viman Nagar" },
          { name: "Wakad" },
        ]
      },
      {
        name: "Nagpur",
        cities: [
          { name: "Dharampeth" },
          { name: "Kamptee" },
          { name: "Saoner" },
          { name: "Sitabuldi" },
        ]
      },
      {
        name: "Thane",
        cities: [
          { name: "Kalyan" },
          { name: "Dombivli" },
          { name: "Mira Road" },
          { name: "Navi Mumbai" },
          { name: "Thane" },
          { name: "Ulhasnagar" },
        ]
      },
    ]
  },
  {
    name: "Delhi",
    districts: [
      {
        name: "Central Delhi",
        cities: [
          { name: "Connaught Place" },
          { name: "Daryaganj" },
          { name: "Karol Bagh" },
          { name: "Paharganj" },
        ]
      },
      {
        name: "New Delhi",
        cities: [
          { name: "Chanakyapuri" },
          { name: "Diplomatic Enclave" },
          { name: "Khan Market" },
          { name: "Safdarjung" },
        ]
      },
      {
        name: "North Delhi",
        cities: [
          { name: "Civil Lines" },
          { name: "Model Town" },
          { name: "Rohini" },
          { name: "Shalimar Bagh" },
        ]
      },
      {
        name: "South Delhi",
        cities: [
          { name: "Hauz Khas" },
          { name: "Lajpat Nagar" },
          { name: "Saket" },
          { name: "Vasant Kunj" },
        ]
      },
      {
        name: "East Delhi",
        cities: [
          { name: "Laxmi Nagar" },
          { name: "Mayur Vihar" },
          { name: "Preet Vihar" },
        ]
      },
    ]
  },
  {
    name: "Karnataka",
    districts: [
      {
        name: "Bangalore Urban",
        cities: [
          { name: "Bangalore" },
          { name: "Byatarayanapura" },
          { name: "Anekal" },
        ]
      },
      {
        name: "Mysore",
        cities: [
          { name: "Mysore" },
          { name: "Hunsur" },
          { name: "Nanjangud" },
          { name: "Periyapatna" },
        ]
      },
      {
        name: "Belgaum",
        cities: [
          { name: "Belgaum" },
          { name: "Bailhongal" },
          { name: "Gokak" },
        ]
      },
    ]
  },
  {
    name: "Tamil Nadu",
    districts: [
      {
        name: "Chennai",
        cities: [
          { name: "Adyar" },
          { name: "Anna Nagar" },
          { name: "Guindy" },
          { name: "Mylapore" },
          { name: "T Nagar" },
          { name: "Velachery" },
        ]
      },
      {
        name: "Coimbatore",
        cities: [
          { name: "Coimbatore" },
          { name: "Mettupalayam" },
          { name: "Pollachi" },
        ]
      },
      {
        name: "Madurai",
        cities: [
          { name: "Madurai" },
          { name: "Melur" },
          { name: "Usilampatti" },
        ]
      },
    ]
  },
  {
    name: "Uttar Pradesh",
    districts: [
      {
        name: "Lucknow",
        cities: [
          { name: "Alambagh" },
          { name: "Gomti Nagar" },
          { name: "Hazratganj" },
          { name: "Indira Nagar" },
        ]
      },
      {
        name: "Kanpur",
        cities: [
          { name: "Kanpur" },
          { name: "Ghatampur" },
        ]
      },
      {
        name: "Varanasi",
        cities: [
          { name: "Varanasi" },
          { name: "Cholapur" },
        ]
      },
      {
        name: "Agra",
        cities: [
          { name: "Agra" },
          { name: "Fatehabad" },
          { name: "Kheragarh" },
        ]
      },
    ]
  },
  {
    name: "Gujarat",
    districts: [
      {
        name: "Ahmedabad",
        cities: [
          { name: "Ahmedabad" },
          { name: "Dholka" },
          { name: "Sanand" },
        ]
      },
      {
        name: "Surat",
        cities: [
          { name: "Surat" },
          { name: "Bardoli" },
          { name: "Choryasi" },
        ]
      },
      {
        name: "Vadodara",
        cities: [
          { name: "Vadodara" },
          { name: "Karjan" },
          { name: "Padra" },
        ]
      },
    ]
  },
  {
    name: "Rajasthan",
    districts: [
      {
        name: "Jaipur",
        cities: [
          { name: "Jaipur" },
          { name: "Amber" },
          { name: "Sanganer" },
        ]
      },
      {
        name: "Jodhpur",
        cities: [
          { name: "Jodhpur" },
          { name: "Bilara" },
          { name: "Osian" },
        ]
      },
      {
        name: "Udaipur",
        cities: [
          { name: "Udaipur" },
          { name: "Kherwara" },
          { name: "Mavli" },
        ]
      },
    ]
  },
  {
    name: "Madhya Pradesh",
    districts: [
      {
        name: "Indore",
        cities: [
          { name: "Indore" },
          { name: "Mhow" },
        ]
      },
      {
        name: "Bhopal",
        cities: [
          { name: "Bhopal" },
          { name: "Berasia" },
        ]
      },
      {
        name: "Jabalpur",
        cities: [
          { name: "Jabalpur" },
          { name: "Patan" },
        ]
      },
    ]
  },
  {
    name: "Andhra Pradesh",
    districts: [
      {
        name: "Visakhapatnam",
        cities: [
          { name: "Visakhapatnam" },
          { name: "Anakapalle" },
        ]
      },
      {
        name: "Guntur",
        cities: [
          { name: "Guntur" },
          { name: "Tenali" },
        ]
      },
    ]
  },
  {
    name: "Telangana",
    districts: [
      {
        name: "Hyderabad",
        cities: [
          { name: "Hyderabad" },
          { name: "Secunderabad" },
        ]
      },
      {
        name: "Rangareddy",
        cities: [
          { name: "Shamshabad" },
          { name: "Rajendranagar" },
        ]
      },
    ]
  },
  {
    name: "Kerala",
    districts: [
      {
        name: "Thiruvananthapuram",
        cities: [
          { name: "Thiruvananthapuram" },
          { name: "Neyyattinkara" },
        ]
      },
      {
        name: "Ernakulam",
        cities: [
          { name: "Kochi" },
          { name: "Aluva" },
        ]
      },
      {
        name: "Kozhikode",
        cities: [
          { name: "Kozhikode" },
          { name: "Vadakara" },
        ]
      },
    ]
  },
  {
    name: "Punjab",
    districts: [
      {
        name: "Ludhiana",
        cities: [
          { name: "Ludhiana" },
          { name: "Jagraon" },
        ]
      },
      {
        name: "Amritsar",
        cities: [
          { name: "Amritsar" },
          { name: "Ajnala" },
        ]
      },
    ]
  },
  {
    name: "Haryana",
    districts: [
      {
        name: "Gurgaon",
        cities: [
          { name: "Gurgaon" },
          { name: "Sohna" },
        ]
      },
      {
        name: "Faridabad",
        cities: [
          { name: "Faridabad" },
          { name: "Ballabgarh" },
        ]
      },
    ]
  },
  {
    name: "Bihar",
    districts: [
      {
        name: "Patna",
        cities: [
          { name: "Patna" },
          { name: "Danapur" },
        ]
      },
      {
        name: "Gaya",
        cities: [
          { name: "Gaya" },
          { name: "Bodh Gaya" },
        ]
      },
    ]
  },
  {
    name: "Odisha",
    districts: [
      {
        name: "Khordha",
        cities: [
          { name: "Bhubaneswar" },
          { name: "Jatni" },
        ]
      },
      {
        name: "Cuttack",
        cities: [
          { name: "Cuttack" },
          { name: "Choudwar" },
        ]
      },
    ]
  },
  {
    name: "Jharkhand",
    districts: [
      {
        name: "Ranchi",
        cities: [
          { name: "Ranchi" },
          { name: "Bundu" },
        ]
      },
      {
        name: "Dhanbad",
        cities: [
          { name: "Dhanbad" },
          { name: "Jharia" },
        ]
      },
    ]
  },
  {
    name: "Assam",
    districts: [
      {
        name: "Kamrup Metropolitan",
        cities: [
          { name: "Guwahati" },
          { name: "Dispur" },
        ]
      },
      {
        name: "Dibrugarh",
        cities: [
          { name: "Dibrugarh" },
          { name: "Naharkatiya" },
        ]
      },
    ]
  },
  {
    name: "Chhattisgarh",
    districts: [
      {
        name: "Raipur",
        cities: [
          { name: "Raipur" },
          { name: "Arang" },
        ]
      },
    ]
  },
  {
    name: "Uttarakhand",
    districts: [
      {
        name: "Dehradun",
        cities: [
          { name: "Dehradun" },
          { name: "Mussoorie" },
        ]
      },
    ]
  },
  {
    name: "Himachal Pradesh",
    districts: [
      {
        name: "Shimla",
        cities: [
          { name: "Shimla" },
          { name: "Theog" },
        ]
      },
    ]
  },
  {
    name: "Jammu and Kashmir",
    districts: [
      {
        name: "Srinagar",
        cities: [
          { name: "Srinagar" },
          { name: "Budgam" },
        ]
      },
    ]
  },
  {
    name: "Goa",
    districts: [
      {
        name: "North Goa",
        cities: [
          { name: "Panaji" },
          { name: "Mapusa" },
        ]
      },
      {
        name: "South Goa",
        cities: [
          { name: "Margao" },
          { name: "Vasco da Gama" },
        ]
      },
    ]
  },
];

// Helper function to get all cities from all states
export const getAllCities = (): string[] => {
  const cities: string[] = [];
  indianStates.forEach(state => {
    state.districts.forEach(district => {
      district.cities.forEach(city => {
        cities.push(city.name);
      });
    });
  });
  return cities.sort();
};

// Helper function to search locations by prefix
export const searchLocationsByPrefix = (prefix: string, type: 'state' | 'district' | 'city'): string[] => {
  const normalizedPrefix = prefix.toLowerCase().trim();
  if (!normalizedPrefix) return [];

  const results: string[] = [];

  if (type === 'state') {
    indianStates.forEach(state => {
      if (state.name.toLowerCase().startsWith(normalizedPrefix)) {
        results.push(state.name);
      }
    });
  } else if (type === 'district') {
    indianStates.forEach(state => {
      state.districts.forEach(district => {
        if (district.name.toLowerCase().startsWith(normalizedPrefix)) {
          results.push(district.name);
        }
      });
    });
  } else if (type === 'city') {
    indianStates.forEach(state => {
      state.districts.forEach(district => {
        district.cities.forEach(city => {
          if (city.name.toLowerCase().startsWith(normalizedPrefix)) {
            results.push(city.name);
          }
        });
      });
    });
  }

  return results.sort();
};

// Helper function to get districts by state
export const getDistrictsByState = (stateName: string): string[] => {
  const state = indianStates.find(s => s.name === stateName);
  if (!state) return [];
  return state.districts.map(d => d.name).sort();
};

// Helper function to get cities by state and district
export const getCitiesByDistrict = (stateName: string, districtName: string): string[] => {
  const state = indianStates.find(s => s.name === stateName);
  if (!state) return [];
  const district = state.districts.find(d => d.name === districtName);
  if (!district) return [];
  return district.cities.map(c => c.name).sort();
};

// Validate if location exists in Indian database
export const isValidIndianLocation = (location: string): boolean => {
  const normalizedLocation = location.toLowerCase().trim();
  
  // Check if it's a state
  if (indianStates.some(state => state.name.toLowerCase() === normalizedLocation)) {
    return true;
  }
  
  // Check if it's a district
  for (const state of indianStates) {
    if (state.districts.some(district => district.name.toLowerCase() === normalizedLocation)) {
      return true;
    }
  }
  
  // Check if it's a city
  for (const state of indianStates) {
    for (const district of state.districts) {
      if (district.cities.some(city => city.name.toLowerCase() === normalizedLocation)) {
        return true;
      }
    }
  }
  
  return false;
};
