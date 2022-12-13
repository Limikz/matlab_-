function [theta_res] =  gradient_descent(X, y, alpha, len,lambda)
    theta = zeros(44,1);
    gradient = gradient_function(theta, X, y, len,lambda);
    while sum(abs(gradient),1)>1e-6
        theta = theta - alpha * gradient;
        gradient = gradient_function(theta, X, y, len,lambda);
        disp(sum(abs(gradient),1));
    end
    theta_res = theta;
end