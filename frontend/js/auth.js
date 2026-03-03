// Authentication JavaScript

const API_BASE_URL = 'http://localhost:8000/api/v1';

// Get form elements
const loginForm = document.getElementById('loginForm');
const signupForm = document.getElementById('signupForm');
const showSignupLink = document.getElementById('showSignup');
const showLoginLink = document.getElementById('showLogin');
const authToast = document.getElementById('authToast');

// Toggle between login and signup forms
showSignupLink.addEventListener('click', (e) => {
    e.preventDefault();
    loginForm.style.display = 'none';
    signupForm.style.display = 'block';
});

showLoginLink.addEventListener('click', (e) => {
    e.preventDefault();
    signupForm.style.display = 'none';
    loginForm.style.display = 'block';
});

// Toast notification
function showToast(message, type = 'success') {
    authToast.textContent = message;
    authToast.className = `toast ${type} show`;
    
    setTimeout(() => {
        authToast.classList.remove('show');
    }, 3000);
}

// Validate email
function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

// Validate password
function validatePassword(password) {
    return password.length >= 8;
}

// Login form submission
loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const email = document.getElementById('loginEmail').value.trim();
    const password = document.getElementById('loginPassword').value;
    const remember = loginForm.querySelector('input[name="remember"]').checked;
    
    // Validation
    if (!validateEmail(email)) {
        showToast('Please enter a valid email address', 'error');
        return;
    }
    
    if (!password) {
        showToast('Please enter your password', 'error');
        return;
    }
    
    // Show loading state
    const btn = loginForm.querySelector('.btn-primary');
    const btnText = document.getElementById('loginBtnText');
    const spinner = document.getElementById('loginSpinner');
    
    btn.disabled = true;
    btnText.style.display = 'none';
    spinner.style.display = 'block';
    loginForm.classList.add('loading');
    
    try {
        const response = await fetch(`${API_BASE_URL}/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email, password, remember })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Store token
            if (remember) {
                localStorage.setItem('auth_token', data.token);
                localStorage.setItem('user_data', JSON.stringify(data.user));
            } else {
                sessionStorage.setItem('auth_token', data.token);
                sessionStorage.setItem('user_data', JSON.stringify(data.user));
            }
            
            showToast('Login successful! Redirecting...', 'success');
            
            // Redirect to main app
            setTimeout(() => {
                window.location.href = 'index.html';
            }, 1000);
        } else {
            showToast(data.detail || 'Login failed. Please check your credentials.', 'error');
        }
    } catch (error) {
        console.error('Login error:', error);
        showToast('Network error. Please try again.', 'error');
    } finally {
        btn.disabled = false;
        btnText.style.display = 'block';
        spinner.style.display = 'none';
        loginForm.classList.remove('loading');
    }
});

// Signup form submission
signupForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const name = document.getElementById('signupName').value.trim();
    const email = document.getElementById('signupEmail').value.trim();
    const password = document.getElementById('signupPassword').value;
    const confirmPassword = document.getElementById('signupConfirmPassword').value;
    const termsAccepted = signupForm.querySelector('input[name="terms"]').checked;
    
    // Validation
    if (!name || name.length < 2) {
        showToast('Please enter your full name', 'error');
        return;
    }
    
    if (!validateEmail(email)) {
        showToast('Please enter a valid email address', 'error');
        return;
    }
    
    if (!validatePassword(password)) {
        showToast('Password must be at least 8 characters long', 'error');
        return;
    }
    
    if (password !== confirmPassword) {
        showToast('Passwords do not match', 'error');
        return;
    }
    
    if (!termsAccepted) {
        showToast('Please accept the terms of service', 'error');
        return;
    }
    
    // Show loading state
    const btn = signupForm.querySelector('.btn-primary');
    const btnText = document.getElementById('signupBtnText');
    const spinner = document.getElementById('signupSpinner');
    
    btn.disabled = true;
    btnText.style.display = 'none';
    spinner.style.display = 'block';
    signupForm.classList.add('loading');
    
    try {
        const response = await fetch(`${API_BASE_URL}/auth/signup`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name, email, password })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showToast('Account created successfully! Please log in.', 'success');
            
            // Switch to login form
            setTimeout(() => {
                signupForm.style.display = 'none';
                loginForm.style.display = 'block';
                document.getElementById('loginEmail').value = email;
                signupForm.reset();
            }, 1500);
        } else {
            showToast(data.detail || 'Signup failed. Please try again.', 'error');
        }
    } catch (error) {
        console.error('Signup error:', error);
        showToast('Network error. Please try again.', 'error');
    } finally {
        btn.disabled = false;
        btnText.style.display = 'block';
        spinner.style.display = 'none';
        signupForm.classList.remove('loading');
    }
});

// Real-time password validation
document.getElementById('signupPassword').addEventListener('input', (e) => {
    const password = e.target.value;
    const input = e.target;
    
    if (password.length === 0) {
        input.classList.remove('error', 'success');
    } else if (validatePassword(password)) {
        input.classList.remove('error');
        input.classList.add('success');
    } else {
        input.classList.remove('success');
        input.classList.add('error');
    }
});

// Real-time email validation
document.getElementById('signupEmail').addEventListener('blur', (e) => {
    const email = e.target.value.trim();
    const input = e.target;
    
    if (email.length === 0) {
        input.classList.remove('error', 'success');
    } else if (validateEmail(email)) {
        input.classList.remove('error');
        input.classList.add('success');
    } else {
        input.classList.remove('success');
        input.classList.add('error');
    }
});

// Confirm password validation
document.getElementById('signupConfirmPassword').addEventListener('input', (e) => {
    const password = document.getElementById('signupPassword').value;
    const confirmPassword = e.target.value;
    const input = e.target;
    
    if (confirmPassword.length === 0) {
        input.classList.remove('error', 'success');
    } else if (password === confirmPassword) {
        input.classList.remove('error');
        input.classList.add('success');
    } else {
        input.classList.remove('success');
        input.classList.add('error');
    }
});

// Check if already logged in
function checkAuth() {
    const token = localStorage.getItem('auth_token') || sessionStorage.getItem('auth_token');
    if (token) {
        // Redirect to main app if already logged in
        window.location.href = 'index.html';
    }
}

checkAuth();
