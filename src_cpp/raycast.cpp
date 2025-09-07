#include "../headers_cpp/raycast.h"

#ifndef CUSTOM_PI
#define CUSTOM_PI
const double PI = 3.141592653589793238462643383279502884;
#endif



double Shue97(double r_0, double alpha_0, double one_plus_cos_theta)
{
    return r_0 * std::pow( 2.0/one_plus_cos_theta, alpha_0 );
}


double get_bowshock_radius(  const Point& projection,
                            const Matrix& Rho, const Point& earth_pos,
                            double dr,
                            bool* is_at_bounds )
{
    double bow_r = 0.0;
    double r = 0.0;
    Point p = r * projection + earth_pos;

    double min_value = 0.0;

    if ( Rho.is_point_OOB(p) ) { std::cout << "ERROR: earth position is OOB in get bowshock radius\n"; exit(1); }
    double previous_rho = Rho(p, 0);

    r += dr;
    p = r * projection + earth_pos;

    while ( !Rho.is_point_OOB(p) )
    {
        double rho = Rho(p, 0);
        double value = (rho - previous_rho) * r*r*r*r; 
        // bit horrifying, but this is to try and ignore things near the earth

        if ( value < min_value )
        {
            min_value = value;
            bow_r = r;
        }

        r += dr;
        p = r * projection + earth_pos;

        previous_rho = rho;
    }

    if ( is_at_bounds != nullptr && std::abs( r-dr-bow_r ) < 0.1*dr ) *is_at_bounds = true;
    
    return bow_r;
}


double get_bowshock_radius(  double theta, double phi,
                            const Matrix& Rho, const Point& earth_pos,
                            double dr,
                            bool* is_at_bounds )
{
    double sin_theta = std::sin(theta);

    Point proj = Point(
        -std::cos(theta),
        sin_theta*std::sin(phi),
        sin_theta*std::cos(phi)
    );

    return get_bowshock_radius(proj, Rho, earth_pos, dr, is_at_bounds);
}


inline void squeeze_vector( std::vector<Point>& points )
{
    Point empty_p{};
    points.erase(std::remove(points.begin(), points.end(), empty_p), points.end());
}


std::vector<Point> get_bowshock( const Matrix& Rho, const Point& earth_pos, double dr, int nb_phi, int max_nb_theta, bool is_squeezed )
{
    std::vector<Point> bs_points( nb_phi*max_nb_theta );

    double shue97_radii[max_nb_theta];

    double dphi = 2.0*PI / nb_phi;
    double dtheta = PI / max_nb_theta;

    double theta1 = 0.0;
    for (int itheta=0; itheta<max_nb_theta; itheta++)
    {
        shue97_radii[itheta] = Shue97(5.0, 0.5, 1.0+std::cos(theta1));
        theta1 += dtheta;
    }

    #pragma omp parallel for
    for (int iphi=0; iphi<nb_phi; iphi++)
    {
        double phi = iphi*dphi - PI;
        double sin_phi = std::sin(phi);
        double cos_phi = std::cos(phi);

        double theta2 = 0.0;

        bool is_at_bounds = false;

        for (int itheta=0; itheta<max_nb_theta; itheta++)
        {
            double sin_theta = std::sin(theta2);

            Point proj = Point(
                -std::cos(theta2),
                sin_theta*sin_phi,
                sin_theta*cos_phi
            );

            double r = get_bowshock_radius(proj, Rho, earth_pos, dr, &is_at_bounds);

            if ( is_at_bounds ) break;

            if ( r > shue97_radii[itheta] )
                bs_points[itheta*nb_phi + iphi] = Point(theta2, phi, r);

            theta2 += dtheta;
        }
    }

    if (is_squeezed) squeeze_vector(bs_points);

    return bs_points;
}





void interest_points_helper(    double r_0, double alpha_0, 
                                std::vector<double>* interest_points,
                                const Point* const unsqueezed_bow_shock,
                                const Matrix& J_norm, const Point& earth_pos, 
                                double theta_min,
                                int nb_theta, int nb_phi, 
                                double dr, double dtheta, double dphi,
                                int itheta_start, int itheta_end )
{
    double theta = theta_min + (itheta_start+1)*dtheta;

    for (int itheta=itheta_start; itheta<itheta_end; itheta++)
    {
        theta += dtheta;

	    // if (std::abs(theta) < 0.05) continue;

        double sin_theta = std::sin(theta);
        double cos_theta = std::cos(theta);

        if ( 1 + cos_theta < 1e-3 ) continue;

        double initial_r = Shue97(r_0, alpha_0, 1 + cos_theta);
	    // double final_r = Shue97(12, 1, 1 + cos_theta);
        double non_bs_final_r = Shue97(10.0*r_0, 1, 1 + cos_theta);

        double phi = -PI;

        for (int iphi=0; iphi<nb_phi; iphi++)
        {
            phi += dphi;

            Point bs = unsqueezed_bow_shock[itheta*nb_phi+iphi];

            double final_r;
            if ( bs.z < initial_r ) final_r = non_bs_final_r;
            else final_r = bs.z - 1.0;  //  get radius of the bowshock at (theta,phi) and add a bit of extra space to be sure not to get the bowshock
            //                                                                            -> NOT SURE THIS IS A GOOD IDEA

            double max_value = 0;
            double max_r = initial_r;
            double r = max_r;

            Point proj = Point(
                -cos_theta,
                sin_theta*std::sin(phi),
                sin_theta*std::cos(phi)
            );

            Point p = r * proj + earth_pos;

            while ( r <= final_r && !J_norm.is_point_OOB(p) )
            {
                double value = J_norm(p, 0); // interpolate( p, J_norm, 0 );

                if ( value > max_value )
                {
                    max_value = value;
                    max_r = r;
                }

                r += dr;
                p = r * proj + earth_pos;
            }

            if ( max_r == initial_r ) continue;

            interest_points[ itheta*nb_phi + iphi ].push_back( max_r );
        }
    }
}




double get_median( std::vector<double>& vec )
{
    std::sort( vec.begin(), vec.end() );

    return vec[ vec.size() / 2 ];
}


double get_std_dev( std::vector<double>& vec )
{
    double avg = std::accumulate( vec.begin(), vec.end(), 0.0 ) / vec.size();

    for (double& val: vec) val = (val-avg)*(val-avg);
    
    return std::sqrt( std::accumulate( vec.begin(), vec.end(), 0.0 ) / vec.size() );
}







InterestPoint* get_interest_points( const Matrix& J_norm, const Point& earth_pos,
                                    const Point* const unsqueezed_bow_shock,
                                    double theta_min, double theta_max, 
                                    int nb_theta, int nb_phi, 
                                    double dx, double dr,
                                    double alpha_0_min, double alpha_0_max, int nb_alpha_0,
                                    double r_0_mult_min, double r_0_mult_max, int nb_r_0,
                                    double* p_avg_std_dev )
{
    std::vector<double>* interest_radii_candidates = new std::vector<double>[ nb_theta*nb_phi ];
    if (interest_radii_candidates == nullptr) { std::cout << "ERROR: out of memory when allocating interest radii candidates.\n"; exit(1); }

    double x = earth_pos.x;

    if (p_avg_std_dev) *p_avg_std_dev = 0;
    
    if ( J_norm.is_point_OOB(earth_pos) ) { std::cout << "ERROR: earth position OOB in get interest points.\n"; exit(1); }
    while ( x>0 and x<J_norm.get_shape().x )
    {
        if ( J_norm(x, earth_pos.y, earth_pos.z, 0) > 1e-18 ) break;
        x += dx;
    }

    double r_inner = x - earth_pos.x;       
    double dr_0_mult = (r_0_mult_max - r_0_mult_min) / nb_r_0;
    double dalpha_0 = (alpha_0_max - alpha_0_min) / nb_alpha_0;

    double dtheta = (theta_max - theta_min) / nb_theta;
    double dphi = 2.0*PI / nb_phi;

    const unsigned int nb_threads = std::thread::hardware_concurrency();

    for (double r_0_mult=r_0_mult_min; r_0_mult<=r_0_mult_max; r_0_mult+=dr_0_mult) for (double alpha_0=alpha_0_min; alpha_0<=alpha_0_max; alpha_0+=dalpha_0)
    {
        std::thread t[nb_threads];

        for (int i=0; i<nb_threads; i++)
        {
            int start_index = (i*nb_theta)/nb_threads;
            int end_index = ((i+1)*nb_theta)/nb_threads;

            t[i] = std::thread(
                &interest_points_helper, 
                r_0_mult * r_inner, alpha_0,
                std::ref(interest_radii_candidates), std::ref(unsqueezed_bow_shock), 
                std::ref(J_norm), std::ref(earth_pos),
                theta_min, 
                nb_theta, nb_phi, 
                dr, dtheta, dphi,
                start_index, end_index
            );
        }

        for (int i=0; i<nb_threads; i++) t[i].join();
    }


    InterestPoint* interest_points = new InterestPoint[ nb_theta*nb_phi ];
    if (interest_points == nullptr) { std::cout << "ERROR: out of memory when allocating interest points.\n"; exit(1); }
    double theta = theta_min;

    for (int itheta=0; itheta<nb_theta; itheta++)
    {
        theta += dtheta;
        double phi = -PI;

//        double sin_theta = std::abs( std::sin(theta) );

        for (int iphi=0; iphi<nb_phi; iphi++)
        {
            phi += dphi;

            if ( interest_radii_candidates[ itheta*nb_phi + iphi ].size() == 0 ) 
            {
                interest_points[ itheta*nb_phi + iphi ] = InterestPoint( theta, phi );
                continue;
            }

            double interest_radius = get_median( interest_radii_candidates[ itheta*nb_phi + iphi ] );
            double std_dev = get_std_dev( interest_radii_candidates[ itheta*nb_phi + iphi ] );

            if (p_avg_std_dev) *p_avg_std_dev += std_dev;

            // double weight = std::exp( -std_dev );  // TODO: change weights
            double weight = 1.0 / (1.0 + std_dev);
            // double weight = 5.0 / (5.0 + std_dev*std_dev);
            // if (std::abs(theta) < 0.5*PI) weight *= sin_theta;

            interest_points[ itheta*nb_phi + iphi ] = InterestPoint( theta, phi, interest_radius, weight ); 
        }
    }

    if (p_avg_std_dev) *p_avg_std_dev /= nb_theta*nb_phi;

    delete[] interest_radii_candidates;

    return interest_points;
}

InterestPoint* get_interest_points( const Matrix& J_norm, const Point& earth_pos,
                                    const Matrix& Rho,
                                    double theta_min, double theta_max, 
                                    int nb_theta, int nb_phi, 
                                    double dx, double dr,
                                    double alpha_0_min, double alpha_0_max, int nb_alpha_0,
                                    double r_0_mult_min, double r_0_mult_max, int nb_r_0,
                                    double* p_avg_std_dev )
{
    std::vector<Point> bow_shock = get_bowshock( Rho, earth_pos, dr, nb_phi, nb_theta, false );

    return get_interest_points( J_norm, earth_pos, bow_shock.data(), theta_min, theta_max, nb_theta, nb_phi, dx, dr, alpha_0_min, alpha_0_max, nb_alpha_0, r_0_mult_min, r_0_mult_max, nb_r_0, p_avg_std_dev);
}





void process_interest_points(   InterestPoint* interest_points, 
                                int nb_theta, int nb_phi, 
                                const Shape& shape_sim, const Shape& shape_real,
                                const Point& earth_pos_sim, const Point& earth_pos_real )
{
    // std::cout << shape_sim << std::endl;
    // std::cout << shape_real << std::endl;
    // std::cout << earth_pos_sim << std::endl;
    // std::cout << earth_pos_real << std::endl;

    for (int itheta=0; itheta<nb_theta; itheta++) for (int iphi=0; iphi<nb_phi; iphi++)
    {
        Point point;

        double sin_theta = std::sin(interest_points[itheta*nb_phi + iphi].theta);

        point.x = std::cos(interest_points[itheta*nb_phi + iphi].theta);
        point.y = sin_theta * std::sin(interest_points[itheta*nb_phi + iphi].phi);
        point.z = sin_theta * std::cos(interest_points[itheta*nb_phi + iphi].phi);

        point *= interest_points[itheta*nb_phi + iphi].radius;
        point += earth_pos_sim;

        point.x *= double(shape_real.x) / double(shape_sim.x);
        point.y *= double(shape_real.y) / double(shape_sim.y);
        point.z *= double(shape_real.z) / double(shape_sim.z);

        // std::cout << point << std::endl;

        point -= earth_pos_real;

        interest_points[itheta*nb_phi + iphi].radius = point.norm();
        interest_points[itheta*nb_phi + iphi].theta = std::acos( point.x / std::max(0.1, interest_points[itheta*nb_phi + iphi].radius) );
        interest_points[itheta*nb_phi + iphi].phi = std::acos( point.z / std::max(0.1, std::sqrt( point.y*point.y + point.z*point.z )) );
        interest_points[itheta*nb_phi + iphi].phi *= (point.y>0) - (point.y<=0);
    }
}






void save_interest_points( const std::string& filename, const InterestPoint* interest_points, int nb_theta, int nb_phi )
{
    std::ofstream fs;
    fs.open(filename);

    for (int itheta=0; itheta<nb_theta; itheta++) for (int iphi=0; iphi<nb_phi; iphi++)
    {
        fs  << interest_points[ itheta*nb_phi + iphi ].theta << ','
            << interest_points[ itheta*nb_phi + iphi ].phi << ','
            << interest_points[ itheta*nb_phi + iphi ].radius << ','
            << interest_points[ itheta*nb_phi + iphi ].weight << '\n';
    }

    fs.close();
}




void process_points(    std::vector<Point>& points, 
                        const Shape& shape_sim, const Shape& shape_real,
                        const Point& earth_pos_sim, const Point& earth_pos_real )
{
    for (Point& p: points)
    {
        Point point;

        double sin_theta = std::sin(p.x);

        point.x = std::cos(p.x);
        point.y = sin_theta * std::sin(p.y);
        point.z = sin_theta * std::cos(p.y);

        point *= p.z;
        point += earth_pos_sim;

        point.x *= double(shape_real.x) / double(shape_sim.x);
        point.y *= double(shape_real.y) / double(shape_sim.y);
        point.z *= double(shape_real.z) / double(shape_sim.z);

        // std::cout << point << std::endl;

        point -= earth_pos_real;

        p.z = point.norm();
        p.x = std::acos( point.x / std::max(0.1, p.z) );
        p.y = std::acos( point.z / std::max(0.1, std::sqrt( point.y*point.y + point.z*point.z )) );
        p.y *= (point.y>0) - (point.y<=0);
    }
}






void save_points( const std::string& filename, const std::vector<Point>& points )
{
    std::ofstream fs;
    fs.open(filename);

    for (const Point& p: points)
    {
        fs  << p.x << ','
            << p.y << ','
            << p.z << '\n';
    }

    fs.close();
}
