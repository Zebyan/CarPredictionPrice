import { useState, type ChangeEvent } from 'react';
import { AlertCircle, TrendingUp, Loader } from 'lucide-react';
import './CarPricePredictor.css';

interface FormData {
  marca: string;
  model: string;
  an_fabricatie: number;
  rulaj: number;
  putere: number;
  capacitate_motor: number;
  combustibil: string;
  caroserie: string;
  culoare: string;
  cutie_viteza: string;
}

interface PredictionResponse {
  predicted: number;
  min_price: number;
  max_price: number;
  margin: number;
  confidence: number;
  residual_std: number;
}

/**
 * Brand → models mapping.
 * Keys are EXACTLY like in your Python list:
 * ['audi','bmw','chevrolet','citroen','dacia','fiat','ford','honda','hyundai',
 *  'kia','mazda','mercedes-benz','mitsubishi','nissan','opel','peugeot',
 *  'porche','renault','seat','skoda','suzuki','tesla','toyota','volkswagen','volvo']
 */
const BRAND_MODELS: Record<string, string[]> = {
  audi: [
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8',
    'Q2', 'Q3', 'Q5', 'Q7', 'Q8',
    'TT', 'R8'
  ],

  bmw: [
    '116d', '118d', '118i', '120d', '120i',
    '316d', '318d', '320d', '320i', '325d', '330d', '330i',
    '520d', '520i', '525d', '530d', '530i',
    '730d', '730i', '740d', '740i',
    'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7',
    'i3', 'i4', 'i8'
  ],

  chevrolet: [
    'Spark', 'Matiz', 'Aveo', 'Kalos', 'Lacetti',
    'Cruze', 'Epica', 'Captiva', 'Orlando', 'Trax'
  ],

  citroen: [
    'C1', 'C2', 'C3', 'C3 Aircross',
    'C4', 'C4 Cactus', 'C4 Picasso', 'C4 Grand Picasso',
    'C5', 'C5 Aircross',
    'Berlingo', 'Jumpy', 'Jumper'
  ],

  dacia: [
    'Logan', 'Logan MCV',
    'Sandero', 'Sandero Stepway',
    'Duster',
    'Lodgy', 'Dokker',
    'Spring', 'Jogger'
  ],

  fiat: [
    'Panda', '500', '500C', '500L', '500X',
    'Grande Punto', 'Punto', 'Punto Evo',
    'Tipo', 'Bravo', 'Linea',
    'Doblo', 'Qubo', 'Fiorino'
  ],

  ford: [
    'Ka', 'Ka+', 'Fiesta',
    'Focus', 'Focus C-Max',
    'Mondeo', 'Fusion',
    'B-Max', 'C-Max', 'S-Max',
    'Kuga', 'EcoSport', 'Edge',
    'Galaxy', 'Transit Connect', 'Ranger', 'Puma'
  ],

  honda: [
    'Jazz', 'Civic', 'Insight',
    'Accord',
    'CR-V', 'HR-V'
  ],

  hyundai: [
    'i10', 'i20', 'i30', 'i40',
    'Accent', 'Elantra',
    'Tucson', 'Santa Fe', 'Kona',
    'ix20', 'ix35'
  ],

  kia: [
    'Picanto', 'Rio', 'Ceed', 'Ceed SW',
    'Carens', 'Cerato',
    'Sportage', 'Sorento',
    'Stonic', 'Niro', 'Optima'
  ],

  mazda: [
    '2', '3', '5', '6',
    'MX-3', 'MX-5',
    'CX-3', 'CX-30', 'CX-5', 'CX-7'
  ],

  'mercedes-benz': [
    'A160', 'A180', 'A200',
    'B160', 'B180', 'B200',
    'C180', 'C200', 'C220', 'C250', 'C300',
    'E200', 'E220', 'E250', 'E300', 'E350',
    'S320', 'S350', 'S400', 'S500',
    'GLA', 'GLB', 'GLC', 'GLE', 'GLS', 'G-Class'
  ],

  mitsubishi: [
    'Colt', 'Lancer', 'Lancer Evolution',
    'ASX', 'Outlander', 'Outlander PHEV',
    'Pajero'
  ],

  nissan: [
    'Micra', 'Note',
    'Almera', 'Tiida',
    'Juke', 'Qashqai', 'X-Trail',
    'Leaf', 'Navara'
  ],

  opel: [
    'Agila', 'Corsa', 'Astra',
    'Meriva', 'Zafira', 'Zafira Tourer',
    'Insignia', 'Vectra',
    'Mokka', 'Mokka X', 'Crossland X', 'Grandland X',
    'Combo', 'Vivaro'
  ],

  peugeot: [
    '107', '108',
    '206', '207', '208',
    '306', '307', '308',
    '406', '407', '408', '508',
    '2008', '3008', '5008',
    'Partner', 'Rifter'
  ],

  porche: [
    'Boxster', 'Cayman',
    '911', 'Panamera',
    'Macan', 'Cayenne'
  ],

  renault: [
    'Twingo', 'Clio', 'Modus',
    'Megane', 'Megane Sedan', 'Megane Estate',
    'Fluence', 'Talisman',
    'Laguna',
    'Captur', 'Kadjar', 'Koleos',
    'Scenic', 'Grand Scenic',
    'Kangoo', 'Trafic'
  ],

  seat: [
    'Mii',
    'Ibiza', 'Cordoba',
    'Leon', 'Toledo', 'Altea',
    'Ateca', 'Arona', 'Tarraco',
    'Alhambra'
  ],

  skoda: [
    'Citigo',
    'Fabia', 'Fabia Combi',
    'Rapid', 'Rapid Spaceback',
    'Octavia', 'Octavia Combi',
    'Superb', 'Superb Combi',
    'Roomster',
    'Karoq', 'Kodiaq', 'Kamiq'
  ],

  suzuki: [
    'Alto', 'Celerio',
    'Swift', 'Splash',
    'SX4', 'SX4 S-Cross',
    'Vitara', 'Grand Vitara',
    'Ignis', 'Jimny'
  ],

  tesla: [
    'Model 3', 'Model S', 'Model X', 'Model Y'
  ],

  toyota: [
    'Aygo', 'Yaris',
    'Corolla', 'Auris', 'Avensis',
    'Prius',
    'C-HR', 'RAV4', 'Highlander',
    'Land Cruiser',
    'Proace'
  ],

  volkswagen: [
    'Up!',
    'Lupo', 'Fox',
    'Polo',
    'Golf', 'Golf Plus', 'Golf Variant',
    'Jetta', 'Bora',
    'Passat', 'Passat Variant', 'Arteon',
    'Touran', 'Sharan',
    'T-Roc', 'Tiguan', 'Tiguan Allspace', 'Touareg',
    'Transporter', 'Caddy'
  ],

  volvo: [
    'C30', 'C70',
    'S40', 'V40',
    'S60', 'V60',
    'S80', 'V70',
    'XC40', 'XC60', 'XC70', 'XC90'
  ],
};


const BRANDS = Object.keys(BRAND_MODELS);

/** Simple label formatting for UI, values stay lowercase for the model */
function brandLabel(brand: string): string {
  if (brand === 'bmw') return 'BMW';
  if (brand === 'kia') return 'Kia';
  if (brand === 'tesla') return 'Tesla';
  if (brand === 'volvo') return 'Volvo';
  if (brand === 'volkswagen') return 'Volkswagen';
  if (brand === 'mercedes-benz') return 'Mercedes-Benz';
  if (brand === 'porche') return 'Porsche'; // human-readable label, value remains "porche"
  // Capitalize first letter, rest lowercase
  return brand.charAt(0).toUpperCase() + brand.slice(1).toLowerCase();
}

const getModelsForBrand = (brand: string) => BRAND_MODELS[brand] ?? [];

export default function CarPricePredictor() {
  const API_BASE_URL = 'http://localhost:8000';

  const [formData, setFormData] = useState<FormData>({
    marca: 'volkswagen', // match training value
    model: 'Golf',
    an_fabricatie: 2018,
    rulaj: 120000,
    putere: 110,
    capacitate_motor: 1600,
    combustibil: 'Diesel',
    caroserie: 'Hatchback',
    culoare: 'Negru',
    cutie_viteza: 'Manuala',
  });

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (
    e: ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]:
        name === 'an_fabricatie' ||
        name === 'rulaj' ||
        name === 'putere' ||
        name === 'capacitate_motor'
          ? (value === '' ? 0 : Number(value))
          : value,
    }));
  };

  const handleBrandChange = (e: ChangeEvent<HTMLSelectElement>) => {
    const newBrand = e.target.value;
    setFormData(prev => {
      const modelsForBrand = getModelsForBrand(newBrand);
      const keepModel = modelsForBrand.includes(prev.model) ? prev.model : '';
      return {
        ...prev,
        marca: newBrand,
        model: keepModel,
      };
    });
  };

  const predictPrice = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      if (!formData.marca || !formData.model || !formData.an_fabricatie) {
        throw new Error('Please fill in brand, model and year');
      }

      const response = await fetch(`${API_BASE_URL}/predict/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `API error: ${response.status}`);
      }

      const data: PredictionResponse = await response.json();
      setPrediction(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(
        `${errorMessage}. Make sure the API is running on ${API_BASE_URL}`
      );
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="cpp-root">
      <div className="cpp-inner">
        {/* Header */}
        <header className="cpp-header">
          <div className="cpp-badge">
            <TrendingUp size={16} />
            <span>Car Price Estimator</span>
          </div>
          <h1>Car Price Predictor</h1>
          <p className="cpp-subtitle">
            Backend: FastAPI · Get estimated price ranges
          </p>
          <p className="cpp-api">
            API URL: <span>{API_BASE_URL}</span>
          </p>
        </header>

        {/* Form card */}
        <section className="cpp-card">
          <div className="cpp-grid">
            {/* Brand (select) */}
            <div className="cpp-field">
              <label>Brand</label>
              <select
                name="marca"
                value={formData.marca}
                onChange={handleBrandChange}
              >
                <option value="">Select brand</option>
                {BRANDS.map(brand => (
                  <option key={brand} value={brand}>
                    {brandLabel(brand)}
                  </option>
                ))}
              </select>
            </div>

            {/* Model (depends on brand) */}
            <div className="cpp-field">
              <label>Model</label>
              <select
                name="model"
                value={formData.model}
                onChange={handleChange}
                disabled={!formData.marca}
              >
                <option value="">
                  {formData.marca ? 'Select model' : 'Select brand first'}
                </option>
                {getModelsForBrand(formData.marca).map(model => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </div>

            <div className="cpp-field">
              <label>Year</label>
              <input
                type="number"
                name="an_fabricatie"
                value={formData.an_fabricatie}
                onChange={handleChange}
                max={new Date().getFullYear()}
              />
            </div>

            <div className="cpp-field">
              <label>Mileage (km)</label>
              <input
                type="number"
                name="rulaj"
                value={formData.rulaj}
                onChange={handleChange}
                
              />
            </div>

            <div className="cpp-field">
              <label>Power (HP)</label>
              <input
                type="number"
                name="putere"
                value={formData.putere}
                onChange={handleChange}
              />
            </div>

            <div className="cpp-field">
              <label>Engine (cc)</label>
              <input
                type="number"
                name="capacitate_motor"
                value={formData.capacitate_motor}
                onChange={handleChange}
              />
            </div>

            <div className="cpp-field">
              <label>Fuel</label>
              <select
                name="combustibil"
                value={formData.combustibil}
                onChange={handleChange}
              >
                <option>Benzina</option>
                <option>Diesel</option>
                <option>Hibrid</option>
                <option>Electrica</option>
                <option>GPL</option>
              </select>
            </div>

            <div className="cpp-field">
              <label>Body Type</label>
              <select
                name="caroserie"
                value={formData.caroserie}
                onChange={handleChange}
              >
                <option>Berlina</option>
                <option>SUV</option>
                <option>Hatchback</option>
                <option>Break</option>
                <option>Coupe</option>
                <option>Cabrio</option>
                <option>Minibus</option>
                <option>Pickup</option>
              </select>
            </div>

            <div className="cpp-field">
              <label>Color</label>
              <select
                name="culoare"
                value={formData.culoare}
                onChange={handleChange}
              >
                <option>Negru</option>
                <option>Alb</option>
                <option>Gri</option>
                <option>Argintiu</option>
                <option>Rosu</option>
                <option>Albastru</option>
              </select>
            </div>

            <div className="cpp-field">
              <label>Transmission</label>
              <select
                name="cutie_viteza"
                value={formData.cutie_viteza}
                onChange={handleChange}
              >
                <option>Manuala</option>
                <option>Automata</option>
              </select>
            </div>
          </div>

          <button
            onClick={predictPrice}
            disabled={loading}
            className="cpp-button"
          >
            {loading ? (
              <>
                <Loader className="cpp-spinner" />
                Predicting…
              </>
            ) : (
              'Predict Price'
            )}
          </button>

          {error && (
            <div className="cpp-error">
              <AlertCircle size={16} />
              <p>{error}</p>
            </div>
          )}
        </section>

        {/* Result card */}
        {prediction && (
          <section className="cpp-card cpp-results">
            <h2>Estimated price</h2>
            <div className="cpp-result-grid">
              <div className="cpp-result-box">
                <p className="cpp-result-label">Minimum</p>
                <p className="cpp-result-value">
                  €{Math.round(prediction.min_price).toLocaleString()}
                </p>
              </div>
              <div className="cpp-result-box cpp-result-main">
                <p className="cpp-result-label">Best estimate</p>
                <p className="cpp-result-value">
                  €{Math.round(prediction.predicted).toLocaleString()}
                </p>
              </div>
              <div className="cpp-result-box">
                <p className="cpp-result-label">Maximum</p>
                <p className="cpp-result-value">
                  €{Math.round(prediction.max_price).toLocaleString()}
                </p>
              </div>
            </div>

            <div className="cpp-meta">
              <p>
                Price spread (±){' '}
                <strong>
                  €{Math.round(prediction.margin).toLocaleString()}
                </strong>
              </p>
              <p>
                Model confidence{' '}
                <strong>{prediction.confidence.toFixed(1)}%</strong>
              </p>
              <p>
                Residual standard deviation{' '}
                <strong>
                  €{Math.round(prediction.residual_std).toLocaleString()}
                </strong>
              </p>
            </div>

            <p className="cpp-footnote">
              This estimate is generated by a machine learning model trained on
              historical listings. Actual sale price can vary with condition,
              service history, negotiation, and local market demand.
            </p>
          </section>
        )}
      </div>
    </div>
  );
}
