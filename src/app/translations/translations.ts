import { Language } from '../context/LanguageContext';

export const translations = {
  en: {
    // Navigation
    nav: {
      home: 'Home',
      assessment: 'Take Assessment',
      learn: 'Learn About TB',
      prevention: 'Prevention',
      findHelp: 'Find Help',
      privacy: 'Privacy'
    },
    // Home Page
    home: {
      badge: 'WHO Aligned',
      title: 'TB Early Risk Detection',
      subtitle: 'AI-Assisted Preliminary Assessment',
      description: 'Get a preliminary tuberculosis risk assessment in minutes. Our intelligent screening tool helps identify potential TB symptoms early.',
      cta: 'Start Free Assessment',
      disclaimer: 'Medical Disclaimer',
      disclaimerText: 'This is a preliminary screening tool only and NOT a diagnostic service. Results do not replace professional medical diagnosis. If you have concerning symptoms, please consult a healthcare provider immediately.',
      features: {
        title: 'How It Works',
        step1: 'Answer Questions',
        step1Desc: 'Complete a brief symptom assessment',
        step2: 'Get Results',
        step2Desc: 'Receive your preliminary risk assessment',
        step3: 'Find Help',
        step3Desc: 'Locate nearby healthcare facilities'
      },
      trust: {
        title: 'Trusted by Healthcare Professionals',
        stat1: '10,000+',
        stat1Label: 'Assessments Completed',
        stat2: '95%',
        stat2Label: 'Accuracy Rate',
        stat3: '500+',
        stat3Label: 'Partner Clinics'
      }
    },
    // Assessment Page
    assessment: {
      stepOf: 'Step',
      of: 'of',
      questions: [
        {
          question: 'Have you had a fever for two weeks or more?',
          description: 'Persistent fever lasting 14 days or longer'
        },
        {
          question: 'Are you coughing up blood?',
          description: 'Blood visible when coughing'
        },
        {
          question: 'Is there blood in your sputum (mucus)?',
          description: 'Blood-streaked mucus when coughing (hemoptysis)'
        },
        {
          question: 'Do you experience night sweats?',
          description: 'Heavy sweating during sleep that soaks bedding'
        },
        {
          question: 'Do you have chest pain while breathing or coughing?',
          description: 'Pain in chest during breathing or coughing'
        },
        {
          question: 'Do you experience shortness of breath?',
          description: 'Difficulty breathing during normal activities'
        },
        {
          question: 'Have you experienced unexplained weight loss?',
          description: 'Unintentional, significant weight loss'
        },
        {
          question: 'Do you feel fatigue and weakness?',
          description: 'Persistent exhaustion and feeling weak'
        },
        {
          question: 'Do you have lumps on your armpits or neck?',
          description: 'Swollen or enlarged areas in armpits or neck'
        },
        {
          question: 'Have you had a cough for 2-4 weeks?',
          description: 'Persistent cough lasting 2 to 4 weeks'
        },
        {
          question: 'Do you have swollen lymph nodes?',
          description: 'Enlarged lymph glands in neck, armpits, or groin'
        },
        {
          question: 'Have you experienced loss of appetite?',
          description: 'Reduced desire to eat, even favorite foods'
        }
      ],
      location: {
        title: 'Where are you located?',
        description: 'This helps us show nearby healthcare facilities',
        district: 'District',
        districtPlaceholder: 'Enter your district',
        state: 'State',
        statePlaceholder: 'Enter your state',
        submit: 'Get My Results'
      },
      buttons: {
        yes: 'Yes',
        no: 'No',
        back: 'Back'
      }
    },
    // Results Page
    results: {
      title: 'Your Assessment Results',
      disclaimer: 'Important Notice',
      disclaimerText: 'This is a preliminary screening only. These results are NOT a diagnosis. Please consult a healthcare professional for proper medical evaluation.',
      riskLevel: 'Risk Level',
      confidence: 'Confidence',
      low: 'Low Risk',
      medium: 'Medium Risk',
      high: 'High Risk',
      lowDesc: 'Based on your responses, you show few TB symptoms. Continue monitoring your health.',
      mediumDesc: 'You have reported some TB symptoms. We recommend consulting a healthcare provider for evaluation.',
      highDesc: 'You have reported multiple TB symptoms. Please seek medical attention as soon as possible.',
      symptomsReported: 'Symptoms Reported',
      symptomsOf: 'of',
      location: 'Your Location',
      nextSteps: 'Recommended Next Steps',
      consultDoctor: 'Consult a Doctor',
      consultDoctorDesc: 'Schedule an appointment with a healthcare provider',
      getTested: 'Get Tested',
      getTestedDesc: 'Visit a clinic for TB screening and diagnosis',
      learnMore: 'Learn More',
      learnMoreDesc: 'Understand TB prevention and treatment',
      findNearby: 'Find Nearby Facilities',
      newAssessment: 'Take New Assessment',
      downloadResults: 'Download Results'
    },
    // Learn Page
    learn: {
      title: 'Learn About Tuberculosis',
      subtitle: 'Understanding TB: Causes, Symptoms, and Treatment',
      whatIs: 'What is Tuberculosis?',
      whatIsDesc: 'Tuberculosis (TB) is a potentially serious infectious disease that mainly affects the lungs. The bacteria that cause tuberculosis are spread from person to person through tiny droplets released into the air via coughs and sneezes.',
      symptoms: 'Common Symptoms',
      symptomsIntro: 'TB symptoms may include:',
      symptom1: 'Persistent cough lasting 3 weeks or more',
      symptom2: 'Coughing up blood or mucus',
      symptom3: 'Chest pain when breathing or coughing',
      symptom4: 'Unintentional weight loss',
      symptom5: 'Fatigue and weakness',
      symptom6: 'Fever and night sweats',
      symptom7: 'Loss of appetite',
      transmission: 'How TB Spreads',
      transmissionDesc: 'TB is spread through the air when a person with active TB disease in their lungs coughs, sneezes, speaks, or sings. People nearby may breathe in these bacteria and become infected.',
      treatment: 'Treatment',
      treatmentDesc: 'TB is treatable and curable. Treatment typically involves taking several antibiotics for 6-9 months. It is very important to complete the full course of treatment.',
      cta: 'Take Assessment Now'
    },
    // Prevention Page
    prevention: {
      title: 'TB Prevention',
      subtitle: 'Protecting Yourself and Others from Tuberculosis',
      intro: 'Prevention is key to stopping the spread of tuberculosis. Here are important steps you can take:',
      vaccine: 'BCG Vaccine',
      vaccineDesc: 'The Bacille Calmette-Guérin (BCG) vaccine can help protect against TB, especially in children. Consult your healthcare provider about vaccination.',
      goodHygiene: 'Practice Good Hygiene',
      goodHygieneDesc: 'Cover your mouth and nose when coughing or sneezing. Wash your hands regularly with soap and water.',
      ventilation: 'Ensure Good Ventilation',
      ventilationDesc: 'Keep living spaces well-ventilated. Open windows to allow fresh air circulation, which can reduce TB transmission.',
      avoidCrowds: 'Avoid Crowded Spaces',
      avoidCrowdsDesc: 'Limit time in crowded, poorly ventilated spaces, especially if you have a weakened immune system.',
      completetreatment: 'Complete Treatment',
      completetreatmentDesc: 'If you have TB, take all medications as prescribed. Incomplete treatment can lead to drug-resistant TB.',
      healthyLifestyle: 'Maintain a Healthy Lifestyle',
      healthyLifestyleDesc: 'Eat nutritious food, exercise regularly, get enough sleep, and avoid smoking to strengthen your immune system.',
      cta: 'Check Your Risk Level'
    },
    // Find Help Page
    findHelp: {
      title: 'Find Healthcare Facilities',
      subtitle: 'Locate TB Testing and Treatment Centers Near You',
      search: 'Search by Location',
      districtPlaceholder: 'Enter district',
      statePlaceholder: 'Enter state',
      searchButton: 'Search Facilities',
      searching: 'Searching...',
      facilities: 'Nearby Facilities',
      noLocation: 'Please enter your location to find nearby facilities',
      emergency: 'Emergency Contacts',
      emergencyNumber: 'TB Helpline',
      emergencyNumberValue: '1800-11-6666',
      nationalHelp: 'National Health Helpline',
      nationalHelpValue: '1800-180-1104'
    },
    // Privacy Page
    privacy: {
      title: 'Privacy Policy',
      subtitle: 'How We Protect Your Information',
      lastUpdated: 'Last Updated',
      intro: 'Your privacy is important to us. This policy explains how we collect, use, and protect your information.',
      dataCollection: 'Data Collection',
      dataCollectionDesc: 'We collect only the information necessary to provide our screening service: symptom responses and general location (district/state). We do not collect personally identifiable information.',
      dataUsage: 'How We Use Your Data',
      dataUsageDesc: 'Your responses are used solely to generate your preliminary risk assessment. Location data helps us suggest nearby healthcare facilities.',
      dataSecurity: 'Data Security',
      dataSecurityDesc: 'We implement industry-standard security measures to protect your data. All data transmission is encrypted.',
      dataRetention: 'Data Retention',
      dataRetentionDesc: 'Assessment data is stored temporarily in your browser and is not sent to our servers. You can clear this data at any time by clearing your browser cache.',
      thirdParty: 'Third-Party Sharing',
      thirdPartyDesc: 'We do not sell, trade, or share your personal information with third parties. Your data remains confidential.',
      yourRights: 'Your Rights',
      yourRightsDesc: 'You have the right to access, correct, or delete your information at any time. Contact us for any privacy-related concerns.',
      contact: 'Contact Us',
      contactDesc: 'If you have questions about this privacy policy, please contact us at privacy@tbdetection.health'
    }
  },
  bn: {
    // Navigation (Bengali)
    nav: {
      home: 'হোম',
      assessment: 'মূল্যায়ন করুন',
      learn: 'টিবি সম্পর্কে জানুন',
      prevention: 'প্রতিরোধ',
      findHelp: 'সাহায্য খুঁজুন',
      privacy: 'গোপনীয়তা'
    },
    // Home Page (Bengali)
    home: {
      badge: 'WHO অনুমোদিত',
      title: 'টিবি প্রাথমিক ঝুঁকি সনাক্তকরণ',
      subtitle: 'AI-সহায়তা প্রাথমিক মূল্যায়ন',
      description: 'মিনিটের মধ্যে যক্ষ্মা ঝুঁকির প্রাথমিক মূল্যায়ন পান। আমাদের বুদ্ধিমান স্ক্রিনিং টুল প্রাথমিকভাবে সম্ভাব্য টিবি লক্ষণ চিহ্নিত করতে সহায়তা করে।',
      cta: 'বিনামূল্যে মূল্যায়ন শুরু করুন',
      disclaimer: 'চিকিৎসা দাবিত্যাগ',
      disclaimerText: 'এটি শুধুমাত্র একটি প্রাথমিক স্ক্রিনিং টুল এবং একটি ডায়াগনস্টিক সেবা নয়। ফলাফল পেশাদার চিকিৎসা নির্ণয়ের প্রতিস্থাপন করে না। যদি আপনার উদ্বেগজনক লক্ষণ থাকে, অনুগ্রহ করে অবিলম্বে একজন স্বাস্থ্যসেবা প্রদানকারীর সাথে পরামর্শ করুন।',
      features: {
        title: 'এটি কিভাবে কাজ করে',
        step1: 'প্রশ্নের উত্তর দিন',
        step1Desc: 'একটি সংক্ষিপ্ত লক্ষণ মূল্যায়ন সম্পূর্ণ করুন',
        step2: 'ফলাফল পান',
        step2Desc: 'আপনার প্রাথমিক ঝুঁকি মূল্যায়ন পান',
        step3: 'সাহায্য খুঁজুন',
        step3Desc: 'কাছাকাছি স্বাস্থ্যসেবা সুবিধা খুঁজুন'
      },
      trust: {
        title: 'স্বাস্থ্যসেবা পেশাদারদের দ্বারা বিশ্বস্ত',
        stat1: '১০,০০০+',
        stat1Label: 'সম্পূর্ণ মূল্যায়ন',
        stat2: '৯৫%',
        stat2Label: 'নির্ভুলতার হার',
        stat3: '৫০০+',
        stat3Label: 'পার্টনার ক্লিনিক'
      }
    },
    // Assessment Page (Bengali)
    assessment: {
      stepOf: 'ধাপ',
      of: 'এর',
      questions: [
        {
          question: 'আপনার কি দুই সপ্তাহ বা তার বেশি সময় ধরে জ্বর আছে?',
          description: 'দীর্ঘস্থায়ী জ্বর ১৪ দিন বা তার বেশি সময় ধরে'
        },
        {
          question: 'আপনি কি রক্ত কাশছেন?',
          description: 'কাশির সময় রক্ত দৃশ্যমান'
        },
        {
          question: 'আপনার কফে (শ্লেষ্মা) কি রক্ত আছে?',
          description: 'কাশির সময় রক্ত-ডোরাকাটা শ্লেষ্মা (হিমপটিসিস)'
        },
        {
          question: 'আপনি কি রাতের ঘাম অনুভব করেন?',
          description: 'ঘুমের সময় প্রচুর ঘাম যা বিছানা ভিজিয়ে দেয়'
        },
        {
          question: 'শ্বাস নেওয়ার সময় বা কাশির সময় কি বুকে ব্যথা হয়?',
          description: 'শ্বাস নেওয়ার বা কাশির সময় বুকে ব্যথা'
        },
        {
          question: 'আপনি কি শ্বাসকষ্ট অনুভব করেন?',
          description: 'স্বাভাবিক কার্যকলাপের সময় শ্বাস নিতে সমস্যা'
        },
        {
          question: 'আপনার কি অব্যাখ্যাত ওজন হ্রাস হয়েছে?',
          description: 'অনিচ্ছাকৃত, উল্লেখযোগ্য ওজন হ্রাস'
        },
        {
          question: 'আপনি কি ক্লান্তি এবং দুর্বলতা অনুভব করেন?',
          description: 'দীর্ঘস্থায়ী ক্লান্তি এবং দুর্বল অনুভব করা'
        },
        {
          question: 'আপনার বগলে বা ঘাড়ে কি গোটা আছে?',
          description: 'বগল বা ঘাড়ে ফোলা বা বড় হওয়া অংশ'
        },
        {
          question: 'আপনার কি ২-৪ সপ্তাহ ধরে কাশি আছে?',
          description: 'দীর্ঘস্থায়ী কাশি ২ থেকে ৪ সপ্তাহ ধরে'
        },
        {
          question: 'আপনার কি ফোলা লিম্ফ নোড আছে?',
          description: 'ঘাড়, বগল বা কুঁচকিতে বর্ধিত লিম্ফ গ্রন্থি'
        },
        {
          question: 'আপনার কি ক্ষুধা হ্রাস হয়েছে?',
          description: 'এমনকি প্রিয় খাবারেও খাওয়ার ইচ্ছা হ্রাস'
        }
      ],
      location: {
        title: 'আপনি কোথায় অবস্থিত?',
        description: 'এটি আমাদের কাছাকাছি স্বাস্থ্যসেবা সুবিধা দেখাতে সাহায্য করে',
        district: 'জেলা',
        districtPlaceholder: 'আপনার জেলা লিখুন',
        state: 'রাজ্য',
        statePlaceholder: 'আপনার রাজ্য লিখুন',
        submit: 'আমার ফলাফল পান'
      },
      buttons: {
        yes: 'হ্যাঁ',
        no: 'না',
        back: 'পিছনে'
      }
    },
    // Results Page (Bengali)
    results: {
      title: 'আপনার মূল্যায়নের ফলাফল',
      disclaimer: 'গুরুত্বপূর্ণ নোটিশ',
      disclaimerText: 'এটি শুধুমাত্র একটি প্রাথমিক স্ক্রিনিং। এই ফলাফল একটি নির্ণয় নয়। সঠিক চিকিৎসা মূল্যায়নের জন্য অনুগ্রহ করে একজন স্বাস্থ্যসেবা পেশাদারের সাথে পরামর্শ করুন।',
      riskLevel: 'ঝুঁকির মাত্রা',
      confidence: 'আত্মবিশ্বাস',
      low: 'কম ঝুঁকি',
      medium: 'মাঝারি ঝুঁকি',
      high: 'উচ্চ ঝুঁকি',
      lowDesc: 'আপনার প্রতিক্রিয়ার উপর ভিত্তি করে, আপনি কয়েকটি টিবি লক্ষণ দেখান। আপনার স্বাস্থ্য পর্যবেক্ষণ চালিয়ে যান।',
      mediumDesc: 'আপনি কিছু টিবি লক্ষণ রিপোর্ট করেছেন। আমরা মূল্যায়নের জন্য একজন স্বাস্থ্যসেবা প্রদানকারীর সাথে পরামর্শ করার সুপারিশ করি।',
      highDesc: 'আপনি একাধিক টিবি লক্ষণ রিপোর্ট করেছেন। যত তাড়াতাড়ি সম্ভব চিকিৎসা সহায়তা নিন।',
      symptomsReported: 'রিপোর্ট করা লক্ষণ',
      symptomsOf: 'এর',
      location: 'আপনার অবস্থান',
      nextSteps: 'প্রস্তাবিত পরবর্তী পদক্ষেপ',
      consultDoctor: 'একজন ডাক্তারের সাথে পরামর্শ করুন',
      consultDoctorDesc: 'একজন স্বাস্থ্যসেবা প্রদানকারীর সাথে একটি অ্যাপয়েন্টমেন্ট নির্ধারণ করুন',
      getTested: 'পরীক্ষা করান',
      getTestedDesc: 'টিবি স্ক্রিনিং এবং নির্ণয়ের জন্য একটি ক্লিনিক পরিদর্শন করুন',
      learnMore: 'আরও জানুন',
      learnMoreDesc: 'টিবি প্রতিরোধ এবং চিকিৎসা বুঝুন',
      findNearby: 'কাছাকাছি সুবিধা খুঁজুন',
      newAssessment: 'নতুন মূল্যায়ন করুন',
      downloadResults: 'ফলাফল ডাউনলোড করুন'
    },
    // Learn Page (Bengali)
    learn: {
      title: 'যক্ষ্মা সম্পর্কে জানুন',
      subtitle: 'টিবি বোঝা: কারণ, লক্ষণ এবং চিকিৎসা',
      whatIs: 'যক্ষ্মা কী?',
      whatIsDesc: 'যক্ষ্মা (টিবি) একটি সম্ভাব্য গুরুতর সংক্রামক রোগ যা প্রধানত ফুসফুসকে প্রভাবিত করে। যক্ষ্মা সৃষ্টিকারী ব্যাকটেরিয়া ব্যক্তি থেকে ব্যক্তিতে কাশি এবং হাঁচির মাধ্যমে বাতাসে ছড়িয়ে পড়া ক্ষুদ্র ফোঁটার মাধ্যমে ছড়ায়।',
      symptoms: 'সাধারণ লক্ষণ',
      symptomsIntro: 'টিবি লক্ষণগুলি অন্তর্ভুক্ত করতে পারে:',
      symptom1: '৩ সপ্তাহ বা তার বেশি সময় ধরে দীর্ঘস্থায়ী কাশি',
      symptom2: 'রক্ত বা শ্লেষ্মা কাশি',
      symptom3: 'শ্বাস নেওয়ার বা কাশির সময় বুকে ব্যথা',
      symptom4: 'অনিচ্ছাকৃত ওজন হ্রাস',
      symptom5: 'ক্লান্তি এবং দুর্বলতা',
      symptom6: 'জ্বর এবং রাতের ঘাম',
      symptom7: 'ক্ষুধা হ্রাস',
      transmission: 'টিবি কিভাবে ছড়ায়',
      transmissionDesc: 'টিবি বাতাসের মাধ্যমে ছড়ায় যখন ফুসফুসে সক্রিয় টিবি রোগ আক্রান্ত ব্যক্তি কাশি, হাঁচি, কথা বলে বা গান করে। কাছাকাছি লোকেরা এই ব্যাকটেরিয়া শ্বাস নিতে পারে এবং সংক্রমিত হতে পারে।',
      treatment: 'চিকিৎসা',
      treatmentDesc: 'টিবি চিকিৎসাযোগ্য এবং নিরাময়যোগ্য। চিকিৎসায় সাধারণত ৬-৯ মাসের জন্য বেশ কয়েকটি অ্যান্টিবায়োটিক নেওয়া জড়িত। চিকিৎসার সম্পূর্ণ কোর্স সম্পূর্ণ করা অত্যন্ত গুরুত্বপূর্ণ।',
      cta: 'এখন মূল্যায়ন করুন'
    },
    // Prevention Page (Bengali)
    prevention: {
      title: 'টিবি প্রতিরোধ',
      subtitle: 'যক্ষ্মা থেকে নিজেকে এবং অন্যদের রক্ষা করা',
      intro: 'যক্ষ্মা ছড়ানো বন্ধ করতে প্রতিরোধ গুরুত্বপূর্ণ। এখানে আপনি নিতে পারেন গুরুত্বপূর্ণ পদক্ষেপ:',
      vaccine: 'BCG টিকা',
      vaccineDesc: 'ব্যাসিল ক্যালমেট-গুয়েরিন (BCG) টিকা টিবির বিরুদ্ধে সুরক্ষায় সাহায্য করতে পারে, বিশেষত শিশুদের মধ্যে। টিকা সম্পর্কে আপনার স্বাস্থ্যসেবা প্রদানকারীর সাথে পরামর্শ করুন।',
      goodHygiene: 'ভাল স্বাস্থ্যবিধি অনুশীলন করুন',
      goodHygieneDesc: 'কাশি বা হাঁচির সময় আপনার মুখ এবং নাক ঢেকে রাখুন। নিয়মিত সাবান এবং জল দিয়ে হাত ধুয়ে নিন।',
      ventilation: 'ভাল বায়ুচলাচল নিশ্চিত করুন',
      ventilationDesc: 'থাকার জায়গা ভালভাবে বায়ুচলাচল করুন। তাজা বাতাস চলাচলের জন্য জানালা খুলুন, যা টিবি সংক্রমণ কমাতে পারে।',
      avoidCrowds: 'ভিড়যুক্ত স্থান এড়িয়ে চলুন',
      avoidCrowdsDesc: 'ভিড়যুক্ত, খারাপ বায়ুচলাচল স্থানে সময় সীমিত করুন, বিশেষত যদি আপনার দুর্বল ইমিউন সিস্টেম থাকে।',
      completetreatment: 'চিকিৎসা সম্পূর্ণ করুন',
      completetreatmentDesc: 'যদি আপনার টিবি থাকে, নির্ধারিত সমস্ত ওষুধ নিন। অসম্পূর্ণ চিকিৎসা ড্রাগ-প্রতিরোধী টিবি হতে পারে।',
      healthyLifestyle: 'একটি স্বাস্থ্যকর জীবনধারা বজায় রাখুন',
      healthyLifestyleDesc: 'পুষ্টিকর খাবার খান, নিয়মিত ব্যায়াম করুন, পর্যাপ্ত ঘুম পান এবং আপনার ইমিউন সিস্টেম শক্তিশালী করতে ধূমপান এড়িয়ে চলুন।',
      cta: 'আপনার ঝুঁকির মাত্রা পরীক্ষা করুন'
    },
    // Find Help Page (Bengali)
    findHelp: {
      title: 'স্বাস্থ্যসেবা সুবিধা খুঁজুন',
      subtitle: 'আপনার কাছাকাছি টিবি পরীক্ষা এবং চিকিৎসা কেন্দ্র খুঁজুন',
      search: 'অবস্থান দ্বারা অনুসন্ধান করুন',
      districtPlaceholder: 'জেলা লিখুন',
      statePlaceholder: 'রাজ্য লিখুন',
      searchButton: 'সুবিধা খুঁজুন',
      searching: 'খোঁজা হচ্ছে...',
      facilities: 'কাছাকাছি সুবিধা',
      noLocation: 'কাছাকাছি সুবিধা খুঁজতে আপনার অবস্থান লিখুন',
      emergency: 'জরুরি যোগাযোগ',
      emergencyNumber: 'টিবি হেল্পলাইন',
      emergencyNumberValue: '১৮০০-১১-৬৬৬৬',
      nationalHelp: 'জাতীয় স্বাস্থ্য হেল্পলাইন',
      nationalHelpValue: '১৮০০-১৮০-১১০৪'
    },
    // Privacy Page (Bengali)
    privacy: {
      title: 'গোপনীয়তা নীতি',
      subtitle: 'আমরা কীভাবে আপনার তথ্য সুরক্ষা করি',
      lastUpdated: 'সর্বশেষ আপডেট',
      intro: 'আপনার গোপনীয়তা আমাদের কাছে গুরুত্বপূর্ণ। এই নীতি ব্যাখ্যা করে যে আমরা কীভাবে আপনার তথ্য সংগ্রহ, ব্যবহার এবং সুরক্ষা করি।',
      dataCollection: 'তথ্য সংগ্রহ',
      dataCollectionDesc: 'আমরা শুধুমাত্র আমাদের স্ক্রিনিং সেবা প্রদান করার জন্য প্রয়োজনীয় তথ্য সংগ্রহ করি: লক্ষণ প্রতিক্রিয়া এবং সাধারণ অবস্থান (জেলা/রাজ্য)। আমরা ব্যক্তিগতভাবে শনাক্তযোগ্য তথ্য সংগ্রহ করি না।',
      dataUsage: 'আমরা কীভাবে আপনার তথ্য ব্যবহার করি',
      dataUsageDesc: 'আপনার প্রতিক্রিয়া শুধুমাত্র আপনার প্রাথমিক ঝুঁকি মূল্যায়ন তৈরি করতে ব্যবহৃত হয়। অবস্থান তথ্য আমাদের কাছাকাছি স্বাস্থ্যসেবা সুবিধা প্রস্তাব করতে সাহায্য করে।',
      dataSecurity: 'তথ্য সুরক্ষা',
      dataSecurityDesc: 'আমরা আপনার তথ্য সুরক্ষা করতে শিল্প-মান সুরক্ষা ব্যবস্থা প্রয়োগ করি। সমস্ত তথ্য সংক্রমণ এনক্রিপ্ট করা হয়।',
      dataRetention: 'তথ্য ধরে রাখা',
      dataRetentionDesc: 'মূল্যায়ন তথ্য আপনার ব্রাউজারে অস্থায়ীভাবে সংরক্ষিত এবং আমাদের সার্ভারে পাঠানো হয় না। আপনি যে কোনও সময় আপনার ব্রাউজার ক্যাশে পরিষ্কার করে এই তথ্য মুছতে পারেন।',
      thirdParty: 'তৃতীয় পক্ষের সাথে শেয়ার করা',
      thirdPartyDesc: 'আমরা তৃতীয় পক্ষের সাথে আপনার ব্যক্তিগত তথ্য বিক্রি, বাণিজ্য বা শেয়ার করি না। আপনার তথ্য গোপনীয় থাকে।',
      yourRights: 'আপনার অধিকার',
      yourRightsDesc: 'আপনার যে কোনও সময় আপনার তথ্য অ্যাক্সেস, সংশোধন বা মুছে ফেলার অধিকার রয়েছে। কোনও গোপনীয়তা-সম্পর্কিত উদ্বেগের জন্য আমাদের সাথে যোগাযোগ করুন।',
      contact: 'আমাদের সাথে যোগাযোগ করুন',
      contactDesc: 'এই গোপনীয়তা নীতি সম্পর্কে আপনার কোনও প্রশ্ন থাকলে, অনুগ্রহ করে privacy@tbdetection.health এ আমাদের সাথে যোগাযোগ করুন'
    }
  },
  hi: {
    // Navigation (Hindi)
    nav: {
      home: 'होम',
      assessment: 'मूल्यांकन करें',
      learn: 'टीबी के बारे में जानें',
      prevention: 'रोकथाम',
      findHelp: 'सहायता खोजें',
      privacy: 'गोपनीयता'
    },
    // Home Page (Hindi)
    home: {
      badge: 'WHO अनुमोदित',
      title: 'टीबी प्रारंभिक जोखिम पहचान',
      subtitle: 'AI-सहायता प्राप्त प्रारंभिक मूल्यांकन',
      description: 'मिनटों में तपेदिक जोखिम का प्रारंभिक मूल्यांकन प्राप्त करें। हमारा बुद्धिमान स्क्रीनिंग टूल संभावित टीबी लक्षणों की जल्दी पहचान करने में मदद करता है।',
      cta: 'निःशुल्क मूल्यांकन शुरू करें',
      disclaimer: 'चिकित्सा अस्वीकरण',
      disclaimerText: 'यह केवल एक प्रारंभिक स्क्रीनिंग टूल है और एक निदान सेवा नहीं है। परिणाम पेशेवर चिकित्सा निदान की जगह नहीं लेते। यदि आपके पास चिंताजनक लक्षण हैं, तो कृपया तुरंत एक स्वास्थ्य सेवा प्रदाता से परामर्श करें।',
      features: {
        title: 'यह कैसे काम करता है',
        step1: 'प्रश्नों के उत्तर दें',
        step1Desc: 'एक संक्षिप्त लक्षण मूल्यांकन पूरा करें',
        step2: 'परिणाम प्राप्त करें',
        step2Desc: 'अपना प्रारंभिक जोखिम मूल्यांकन प्राप्त करें',
        step3: 'सहायता खोजें',
        step3Desc: 'पास की स्वास्थ्य सुविधाएं खोजें'
      },
      trust: {
        title: 'स्वास्थ्य पेशेवरों द्वारा विश्वसनीय',
        stat1: '10,000+',
        stat1Label: 'पूर्ण मूल्यांकन',
        stat2: '95%',
        stat2Label: 'सटीकता दर',
        stat3: '500+',
        stat3Label: 'साझेदार क्लिनिक'
      }
    },
    // Assessment Page (Hindi)
    assessment: {
      stepOf: 'चरण',
      of: 'का',
      questions: [
        {
          question: 'क्या आपको दो सप्ताह या उससे अधिक समय से बुखार है?',
          description: 'लगातार बुखार 14 दिन या उससे अधिक समय तक'
        },
        {
          question: 'क्या आप खून की खांसी कर रहे हैं?',
          description: 'खांसते समय खून दिखाई देना'
        },
        {
          question: 'क्या आपके बलगम (कफ) में खून है?',
          description: 'खांसते समय खून-धारीदार बलगम (हेमोप्टिसिस)'
        },
        {
          question: 'क्या आप रात को पसीना अनुभव करते हैं?',
          description: 'नींद के दौरान भारी पसीना जो बिस्तर को भिगो देता है'
        },
        {
          question: 'क्या सांस लेते या खांसते समय सीने में दर्द होता है?',
          description: 'सांस लेने या खांसने के दौरान सीने में दर्द'
        },
        {
          question: 'क्या आप सांस की तकलीफ अनुभव करते हैं?',
          description: 'सामान्य गतिविधियों के दौरान सांस लेने में कठिनाई'
        },
        {
          question: 'क्या आपका वजन अचानक कम हुआ है?',
          description: 'अनजाने में, महत्वपूर्ण वजन घटना'
        },
        {
          question: 'क्या आप थकान और कमजोरी महसूस करते हैं?',
          description: 'लगातार थकान और कमजोर महसूस करना'
        },
        {
          question: 'क्या आपकी बगल या गर्दन पर गांठें हैं?',
          description: 'बगल या गर्दन में सूजे या बढ़े हुए क्षेत्र'
        },
        {
          question: 'क्या आपको 2-4 सप्ताह से खांसी है?',
          description: 'लगातार खांसी 2 से 4 सप्ताह तक'
        },
        {
          question: 'क्या आपके लिम्फ नोड्स सूजे हुए हैं?',
          description: 'गर्दन, बगल या कमर में बढ़े हुए लिम्फ ग्रंथियां'
        },
        {
          question: 'क्या आपकी भूख कम हुई है?',
          description: 'खाने की इच्छा में कमी, पसंदीदा खाद्य पदार्थों में भी'
        }
      ],
      location: {
        title: 'आप कहाँ स्थित हैं?',
        description: 'यह हमें पास की स्वास्थ्य सुविधाएं दिखाने में मदद करता है',
        district: 'जिला',
        districtPlaceholder: 'अपना जिला दर्ज करें',
        state: 'राज्य',
        statePlaceholder: 'अपना राज्य दर्ज करें',
        submit: 'मेरा परिणाम प्राप्त करें'
      },
      buttons: {
        yes: 'हाँ',
        no: 'नहीं',
        back: 'पीछे'
      }
    },
    // Results Page (Hindi)
    results: {
      title: 'आपके मूल्यांकन के परिणाम',
      disclaimer: 'महत्वपूर्ण सूचना',
      disclaimerText: 'यह केवल एक प्रारंभिक स्क्रीनिंग है। ये परिणाम एक निदान नहीं हैं। उचित चिकित्सा मूल्यांकन के लिए कृपया एक स्वास्थ्य पेशेवर से परामर्श करें।',
      riskLevel: 'जोखिम स्तर',
      confidence: 'विश्वास',
      low: 'कम जोखिम',
      medium: 'मध्यम जोखिम',
      high: 'उच्च जोखिम',
      lowDesc: 'आपकी प्रतिक्रियाओं के आधार पर, आप कुछ टीबी लक्षण दिखाते हैं। अपने स्वास्थ्य की निगरानी जारी रखें।',
      mediumDesc: 'आपने कुछ टीबी लक्षणों की रिपोर्ट की है। हम मूल्यांकन के लिए एक स्वास्थ्य सेवा प्रदाता से परामर्श करने की सलाह देते हैं।',
      highDesc: 'आपने कई टीबी लक्षणों की रिपोर्ट की है। कृपया जल्द से जल्द चिकित्सा सहायता लें।',
      symptomsReported: 'रिपोर्ट किए गए लक्षण',
      symptomsOf: 'का',
      location: 'आपका स्थान',
      nextSteps: 'अनुशंसित अगले कदम',
      consultDoctor: 'एक डॉक्टर से परामर्श करें',
      consultDoctorDesc: 'एक स्वास्थ्य सेवा प्रदाता के साथ एक अपॉइंटमेंट निर्धारित करें',
      getTested: 'परीक्षण करवाएं',
      getTestedDesc: 'टीबी स्क्रीनिंग और निदान के लिए एक क्लिनिक पर जाएं',
      learnMore: 'और जानें',
      learnMoreDesc: 'टीबी रोकथाम और उपचार को समझें',
      findNearby: 'पास की सुविधाएं खोजें',
      newAssessment: 'नया मूल्यांकन करें',
      downloadResults: 'परिणाम डाउनलोड करें'
    },
    // Learn Page (Hindi)
    learn: {
      title: 'तपेदिक के बारे में जानें',
      subtitle: 'टीबी को समझना: कारण, लक्षण और उपचार',
      whatIs: 'तपेदिक क्या है?',
      whatIsDesc: 'तपेदिक (टीबी) एक संभावित गंभीर संक्रामक बीमारी है जो मुख्य रूप से फेफड़ों को प्रभावित करती है। तपेदिक पैदा करने वाले बैक्टीरिया व्यक्ति से व्यक्ति में खांसी और छींक के माध्यम से हवा में छोड़े गए छोटी बूंदों के माध्यम से फैलते हैं।',
      symptoms: 'सामान्य लक्षण',
      symptomsIntro: 'टीबी के लक्षणों में शामिल हो सकते हैं:',
      symptom1: '3 सप्ताह या उससे अधिक समय तक लगातार खांसी',
      symptom2: 'खून या बलगम की खांसी',
      symptom3: 'सांस लेते या खांसते समय सीने में दर्द',
      symptom4: 'अनजाने में वजन घटना',
      symptom5: 'थकान और कमजोरी',
      symptom6: 'बुखार और रात को पसीना',
      symptom7: 'भूख में कमी',
      transmission: 'टीबी कैसे फैलता है',
      transmissionDesc: 'टीबी हवा के माध्यम से फैलता है जब फेफड़ों में सक्रिय टीबी रोग वाला व्यक्ति खांसता है, छींकता है, बोलता है, या गाता है। आस-पास के लोग इन बैक्टीरिया को सांस में ले सकते हैं और संक्रमित हो सकते हैं।',
      treatment: 'उपचार',
      treatmentDesc: 'टीबी उपचार योग्य और इलाज योग्य है। उपचार में आम तौर पर 6-9 महीने के लिए कई एंटीबायोटिक्स लेना शामिल है। उपचार का पूरा कोर्स पूरा करना बहुत महत्वपूर्ण है।',
      cta: 'अब मूल्यांकन करें'
    },
    // Prevention Page (Hindi)
    prevention: {
      title: 'टीबी रोकथाम',
      subtitle: 'तपेदिक से खुद को और दूसरों को बचाना',
      intro: 'तपेदिक के प्रसार को रोकने के लिए रोकथाम महत्वपूर्ण है। यहाँ महत्वपूर्ण कदम हैं जो आप उठा सकते हैं:',
      vaccine: 'BCG टीका',
      vaccineDesc: 'बैसिल कैलमेट-गुएरिन (BCG) टीका टीबी से बचाव में मदद कर सकता है, विशेष रूप से बच्चों में। टीकाकरण के बारे में अपने स्वास्थ्य सेवा प्रदाता से परामर्श करें।',
      goodHygiene: 'अच्छी स्वच्छता का अभ्यास करें',
      goodHygieneDesc: 'खांसते या छींकते समय अपनी नाक और मुंह को ढकें। नियमित रूप से साबुन और पानी से हाथ धोएं।',
      ventilation: 'अच्छी हवा सुनिश्चित करें',
      ventilationDesc: 'रहने की जगहों को अच्छी तरह हवादार रखें। ताजी हवा के संचार के लिए खिड़कियां खोलें, जो टीबी संचरण को कम कर सकता है।',
      avoidCrowds: 'भीड़भाड़ वाली जगहों से बचें',
      avoidCrowdsDesc: 'भीड़भाड़ वाली, खराब हवादार जगहों में समय सीमित करें, खासकर यदि आपकी प्रतिरक्षा प्रणाली कमजोर है।',
      completetreatment: 'उपचार पूरा करें',
      completetreatmentDesc: 'यदि आपको टीबी है, तो सभी दवाएं निर्धारित के अनुसार लें। अधूरा उपचार दवा-प्रतिरोधी टीबी का कारण बन सकता है।',
      healthyLifestyle: 'एक स्वस्थ जीवन शैली बनाए रखें',
      healthyLifestyleDesc: 'पौष्टिक भोजन करें, नियमित रूप से व्यायाम करें, पर्याप्त नींद लें और अपनी प्रतिरक्षा प्रणाली को मजबूत करने के लिए धूम्रपान से बचें।',
      cta: 'अपने जोखिम स्तर की जांच करें'
    },
    // Find Help Page (Hindi)
    findHelp: {
      title: 'स्वास्थ्य सुविधाएं खोजें',
      subtitle: 'अपने पास टीबी परीक्षण और उपचार केंद्र खोजें',
      search: 'स्थान द्वारा खोजें',
      districtPlaceholder: 'जिला दर्ज करें',
      statePlaceholder: 'राज्य दर्ज करें',
      searchButton: 'सुविधाएं खोजें',
      searching: 'खोज रहे हैं...',
      facilities: 'पास की सुविधाएं',
      noLocation: 'पास की सुविधाएं खोजने के लिए कृपया अपना स्थान दर्ज करें',
      emergency: 'आपातकालीन संपर्क',
      emergencyNumber: 'टीबी हेल्पलाइन',
      emergencyNumberValue: '1800-11-6666',
      nationalHelp: 'राष्ट्रीय स्वास्थ्य हेल्पलाइन',
      nationalHelpValue: '1800-180-1104'
    },
    // Privacy Page (Hindi)
    privacy: {
      title: 'गोपनीयता नीति',
      subtitle: 'हम आपकी जानकारी की सुरक्षा कैसे करते हैं',
      lastUpdated: 'अंतिम अपडेट',
      intro: 'आपकी गोपनीयता हमारे लिए महत्वपूर्ण है। यह नीति बताती है कि हम आपकी जानकारी को कैसे एकत्र, उपयोग और सुरक्षित करते हैं।',
      dataCollection: 'डेटा संग्रह',
      dataCollectionDesc: 'हम केवल हमारी स्क्रीनिंग सेवा प्रदान करने के लिए आवश्यक जानकारी एकत्र करते हैं: लक्षण प्रतिक्रियाएं और सामान्य स्थान (जिला/राज्य)। हम व्यक्तिगत रूप से पहचानने योग्य जानकारी एकत्र नहीं करते हैं।',
      dataUsage: 'हम आपके डेटा का उपयोग कैसे करते हैं',
      dataUsageDesc: 'आपकी प्रतिक्रियाओं का उपयोग केवल आपके प्रारंभिक जोखिम मूल्यांकन को उत्पन्न करने के लिए किया जाता है। स्थान डेटा हमें पास की स्वास्थ्य सुविधाओं का सुझाव देने में मदद करता है।',
      dataSecurity: 'डेटा सुरक्षा',
      dataSecurityDesc: 'हम आपके डेटा की सुरक्षा के लिए उद्योग-मानक सुरक्षा उपायों को लागू करते हैं। सभी डेटा संचरण एन्क्रिप्ट किया गया है।',
      dataRetention: 'डेटा प्रतिधारण',
      dataRetentionDesc: 'मूल्यांकन डेटा आपके ब्राउज़र में अस्थायी रूप से संग्रहीत है और हमारे सर्वर पर नहीं भेजा जाता है। आप किसी भी समय अपने ब्राउज़र कैश को साफ़ करके इस डेटा को हटा सकते हैं।',
      thirdParty: 'तृतीय-पक्ष साझाकरण',
      thirdPartyDesc: 'हम तृतीय पक्षों के साथ आपकी व्यक्तिगत जानकारी को बेचते, व्यापार या साझा नहीं करते हैं। आपका डेटा ग���पनीय रहता है।',
      yourRights: 'आपके अधिकार',
      yourRightsDesc: 'आपको किसी भी समय अपनी जानकारी तक पहुंचने, सही करने या हटाने का अधिकार है। किसी भी गोपनीयता-संबंधित चिंताओं के लिए हमसे संपर्क करें।',
      contact: 'हमसे संपर्क करें',
      contactDesc: 'यदि इस गोपनीयता नीति के बारे में आपके कोई प्रश्न हैं, तो कृपया हमसे privacy@tbdetection.health पर संपर्क करें'
    }
  }
};

export function getTranslation(language: Language) {
  return translations[language];
}
