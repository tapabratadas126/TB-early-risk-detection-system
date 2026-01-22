// Hospital and Healthcare Facilities Database for TB Treatment in India

export interface Hospital {
  name: string;
  address: string;
  phone: string;
  hours: string;
  website?: string;
  district: string;
  state: string;
}

export const hospitalDatabase: Hospital[] = [
  // Delhi
  {
    name: 'All India Institute of Medical Sciences (AIIMS)',
    address: 'Ansari Nagar, New Delhi',
    phone: '+91-11-26588500',
    hours: '24/7 Emergency Services',
    website: 'https://www.aiims.edu',
    district: 'New Delhi',
    state: 'Delhi'
  },
  {
    name: 'Lala Ram Sarup Institute of Tuberculosis and Respiratory Diseases',
    address: 'Mehrauli, New Delhi',
    phone: '+91-11-26177909',
    hours: 'Mon-Sat: 9:00 AM - 5:00 PM',
    website: 'https://lrsitbrd.delhi.gov.in',
    district: 'South Delhi',
    state: 'Delhi'
  },
  {
    name: 'Rajan Babu Institute for Pulmonary Medicine and Tuberculosis',
    address: 'GTB Nagar, Delhi',
    phone: '+91-11-27667314',
    hours: 'Mon-Sat: 8:00 AM - 6:00 PM',
    website: 'https://rbipmtb.delhi.gov.in',
    district: 'North Delhi',
    state: 'Delhi'
  },
  
  // Maharashtra
  {
    name: 'B.Y.L. Nair Charitable Hospital',
    address: 'Dr. A.L. Nair Road, Mumbai Central',
    phone: '+91-22-23027643',
    hours: '24/7 Emergency Services',
    website: 'https://www.nairhos pital.in',
    district: 'Mumbai City',
    state: 'Maharashtra'
  },
  {
    name: 'Sewri TB Hospital',
    address: 'Sewri, Mumbai',
    phone: '+91-22-24143283',
    hours: 'Mon-Sat: 9:00 AM - 5:00 PM',
    website: 'https://www.sewritbhospital.in',
    district: 'Mumbai City',
    state: 'Maharashtra'
  },
  {
    name: 'Aundh Chest Hospital',
    address: 'Aundh, Pune',
    phone: '+91-20-25886172',
    hours: 'Mon-Sat: 8:00 AM - 6:00 PM',
    website: 'https://www.punechesthospital.org',
    district: 'Pune',
    state: 'Maharashtra'
  },
  
  // West Bengal
  {
    name: 'Institute of Pulmocare and Research',
    address: 'Salt Lake, Kolkata',
    phone: '+91-33-23343106',
    hours: '24/7 Emergency Services',
    website: 'https://www.pulmocare.org',
    district: 'Kolkata',
    state: 'West Bengal'
  },
  {
    name: 'B.C. Roy Memorial Hospital for Children',
    address: 'Phoolbagan, Kolkata',
    phone: '+91-33-22411645',
    hours: 'Mon-Sat: 9:00 AM - 5:00 PM',
    website: 'https://www.bcroyhospital.org',
    district: 'Kolkata',
    state: 'West Bengal'
  },
  
  // Tamil Nadu
  {
    name: 'Government Hospital of Thoracic Medicine',
    address: 'Tambaram, Chennai',
    phone: '+91-44-22262231',
    hours: '24/7 Emergency Services',
    website: 'https://www.ghtm.in',
    district: 'Chennai',
    state: 'Tamil Nadu'
  },
  {
    name: 'Institute of Thoracic Medicine',
    address: 'Chetpet, Chennai',
    phone: '+91-44-28332055',
    hours: 'Mon-Sat: 8:00 AM - 6:00 PM',
    website: 'https://www.itm.tn.gov.in',
    district: 'Chennai',
    state: 'Tamil Nadu'
  },
  
  // Karnataka
  {
    name: 'Rajiv Gandhi Institute of Chest Diseases',
    address: 'Victoria Hospital Campus, Bangalore',
    phone: '+91-80-26702810',
    hours: '24/7 Emergency Services',
    website: 'https://www.rgicd.kar.nic.in',
    district: 'Bengaluru Urban',
    state: 'Karnataka'
  },
  {
    name: 'Bowring and Lady Curzon Hospital',
    address: 'Shivajinagar, Bangalore',
    phone: '+91-80-25211363',
    hours: 'Mon-Sat: 9:00 AM - 5:00 PM',
    website: 'https://www.blch.in',
    district: 'Bengaluru Urban',
    state: 'Karnataka'
  },
  
  // Uttar Pradesh
  {
    name: 'King George\'s Medical University',
    address: 'Shah Mina Road, Lucknow',
    phone: '+91-522-2257450',
    hours: '24/7 Emergency Services',
    website: 'https://www.kgmu.org',
    district: 'Lucknow',
    state: 'Uttar Pradesh'
  },
  {
    name: 'Vallabhbhai Patel Chest Institute',
    address: 'University of Delhi, Delhi',
    phone: '+91-11-27667420',
    hours: 'Mon-Sat: 8:00 AM - 6:00 PM',
    website: 'https://www.vpci.org.in',
    district: 'North Delhi',
    state: 'Delhi'
  },
  
  // Gujarat
  {
    name: 'Civil Hospital Tuberculosis Unit',
    address: 'Asarwa, Ahmedabad',
    phone: '+91-79-22682074',
    hours: 'Mon-Sat: 9:00 AM - 5:00 PM',
    website: 'https://www.ahmedabadcivil.gujarat.gov.in',
    district: 'Ahmedabad',
    state: 'Gujarat'
  },
  
  // Telangana
  {
    name: 'Chest Hospital',
    address: 'Erragadda, Hyderabad',
    phone: '+91-40-23814939',
    hours: '24/7 Emergency Services',
    website: 'https://www.chesthospital.telangana.gov.in',
    district: 'Hyderabad',
    state: 'Telangana'
  },
  
  // Rajasthan
  {
    name: 'SMS Medical College and Hospital',
    address: 'JLN Marg, Jaipur',
    phone: '+91-141-2518121',
    hours: '24/7 Emergency Services',
    website: 'https://www.smsmedicalcollege.com',
    district: 'Jaipur',
    state: 'Rajasthan'
  },
  
  // Bihar
  {
    name: 'Indira Gandhi Institute of Medical Sciences',
    address: 'Sheikhpura, Patna',
    phone: '+91-612-2297089',
    hours: '24/7 Emergency Services',
    website: 'https://www.igims.org',
    district: 'Patna',
    state: 'Bihar'
  },
  
  // Madhya Pradesh
  {
    name: 'All India Institute of Medical Sciences Bhopal',
    address: 'Saket Nagar, Bhopal',
    phone: '+91-755-2672350',
    hours: '24/7 Emergency Services',
    website: 'https://www.aiimsbhopal.edu.in',
    district: 'Bhopal',
    state: 'Madhya Pradesh'
  },
  
  // Punjab
  {
    name: 'Government Medical College and Hospital',
    address: 'Sector 32, Chandigarh',
    phone: '+91-172-2601023',
    hours: '24/7 Emergency Services',
    website: 'https://www.gmch.gov.in',
    district: 'Chandigarh',
    state: 'Chandigarh'
  },
  
  // Assam
  {
    name: 'Gauhati Medical College and Hospital',
    address: 'Bhangagarh, Guwahati',
    phone: '+91-361-2528662',
    hours: '24/7 Emergency Services',
    website: 'https://www.gmch.gov.in',
    district: 'Kamrup Metropolitan',
    state: 'Assam'
  },
  
  // Kerala
  {
    name: 'Government Medical College',
    address: 'Kottayam, Kerala',
    phone: '+91-481-2597275',
    hours: 'Mon-Sat: 8:00 AM - 6:00 PM',
    website: 'https://www.gmckottayam.ac.in',
    district: 'Kottayam',
    state: 'Kerala'
  },
  
  // Odisha
  {
    name: 'SCB Medical College and Hospital',
    address: 'Mangalabag, Cuttack',
    phone: '+91-671-2300884',
    hours: '24/7 Emergency Services',
    website: 'https://www.scbmch.ac.in',
    district: 'Cuttack',
    state: 'Odisha'
  }
];

// National TB Helplines and Important Contacts
export const nationalContacts = [
  {
    name: 'National TB Elimination Programme Helpline',
    phone: '1800-11-6666',
    description: 'Toll-free helpline for TB-related queries'
  },
  {
    name: 'National Health Portal',
    phone: '1800-180-1104',
    description: '24/7 health helpline'
  },
  {
    name: 'Central TB Division, Ministry of Health',
    phone: '+91-11-23061502',
    description: 'Government TB programme information'
  }
];

// Helper functions
export function getHospitalsByDistrict(district: string, state: string): Hospital[] {
  return hospitalDatabase.filter(
    h => h.district.toLowerCase() === district.toLowerCase() && 
         h.state.toLowerCase() === state.toLowerCase()
  );
}

export function getHospitalsByState(state: string): Hospital[] {
  return hospitalDatabase.filter(
    h => h.state.toLowerCase() === state.toLowerCase()
  );
}

export function getNearbyStateHospitals(state: string): Hospital[] {
  // Map of nearby states for fallback
  const nearbyStatesMap: Record<string, string[]> = {
    'Delhi': ['Haryana', 'Uttar Pradesh', 'Punjab'],
    'Maharashtra': ['Gujarat', 'Madhya Pradesh', 'Karnataka'],
    'West Bengal': ['Bihar', 'Jharkhand', 'Odisha', 'Assam'],
    'Tamil Nadu': ['Kerala', 'Karnataka', 'Andhra Pradesh'],
    'Karnataka': ['Maharashtra', 'Telangana', 'Tamil Nadu', 'Kerala'],
    'Uttar Pradesh': ['Delhi', 'Haryana', 'Madhya Pradesh', 'Bihar'],
    'Gujarat': ['Maharashtra', 'Rajasthan', 'Madhya Pradesh'],
    'Telangana': ['Karnataka', 'Maharashtra', 'Andhra Pradesh'],
    'Rajasthan': ['Gujarat', 'Madhya Pradesh', 'Haryana', 'Punjab'],
    'Bihar': ['Uttar Pradesh', 'Jharkhand', 'West Bengal'],
    'Madhya Pradesh': ['Maharashtra', 'Gujarat', 'Rajasthan', 'Uttar Pradesh'],
    'Punjab': ['Haryana', 'Delhi', 'Rajasthan', 'Chandigarh'],
    'Assam': ['West Bengal', 'Meghalaya', 'Arunachal Pradesh'],
    'Kerala': ['Tamil Nadu', 'Karnataka'],
    'Odisha': ['West Bengal', 'Jharkhand', 'Chhattisgarh'],
    'Chandigarh': ['Punjab', 'Haryana', 'Delhi']
  };
  
  const nearbyStates = nearbyStatesMap[state] || [];
  const hospitals: Hospital[] = [];
  
  nearbyStates.forEach(nearbyState => {
    hospitals.push(...getHospitalsByState(nearbyState));
  });
  
  return hospitals.slice(0, 5); // Return top 5 nearby hospitals
}
